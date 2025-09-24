# maskedautoflow/pipeline.py

# =============================================================================
# Masked Autoregressive Flow (MAF) pipeline
# -----------------------------------------------------------------------------
# - Trains a single *unconditional* MAF on flattened images in [0, 1]
# - Saves best/last checkpoints compatible with Keras 3 (save_weights(filepath))
# - Synthesizes a class-balanced set by evenly splitting unconditional samples
# - Produces evaluator-friendly artifacts:
#       gen_class_{k}.npy, labels_class_{k}.npy, x_synth.npy, y_synth.npy
# - Compatible with app/main.py orchestration (train → synth → eval)
#
# Notes
# -----
# • y_* inputs are accepted for API parity but unused (model is unconditional).
# • The preview helper `_sample_batch(...)` accepts **kwargs so callers that
#   pass extra keys (e.g., `bundle=...`) won’t break.
# • A sampling temperature knob (SAMPLE_TEMPERATURE) is provided to tune
#   fidelity/diversity at generation time (τ ∈ (0, ∞), default 1.0).
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, List

import numpy as np
import tensorflow as tf

from maskedautoflow.models import (
    MAF,
    MAFConfig,
    build_maf_model,
    flatten_images,
    reshape_to_images,
)

# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def _ensure_dir(p: Path) -> None:
    """Create directory (parents included) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def _one_hot(y_int: np.ndarray, num_classes: int) -> np.ndarray:
    """Minimal 1-hot encoder. Accepts 1-D int labels."""
    y_int = y_int.ravel().astype(int)
    out = np.zeros((len(y_int), num_classes), dtype=np.float32)
    out[np.arange(len(y_int)), y_int] = 1.0
    return out


# ---------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------
@dataclass
class MAFPipelineConfig:
    # Data / shapes
    IMG_SHAPE: Tuple[int, int, int] = (40, 40, 1)
    NUM_CLASSES: int = 9

    # Model
    NUM_FLOWS: int = 5
    HIDDEN_DIMS: Tuple[int, ...] = (128, 128)

    # Training
    EPOCHS: int = 5
    BATCH_SIZE: int = 256
    LR: float = 2e-4
    CLIP_GRAD: float = 1.0
    PATIENCE: int = 10
    SEED: int = 42

    # Synthesis
    SAMPLES_PER_CLASS: int = 25
    SAMPLE_TEMPERATURE: float = 1.0  # τ=1.0 default; <1 sharper, >1 more diverse

    # Artifacts
    ARTIFACTS: Dict[str, str] = None  # resolved in pipeline __init__


class MAFPipeline:
    """
    Training + synthesis wrapper for an *unconditional* Masked Autoregressive Flow.

    Responsibilities
    ----------------
    • Train a global MAF on flattened images in [0,1]
    • Save best/last checkpoints under ARTIFACTS['checkpoints']
    • Synthesize a class-balanced dataset (simple even split across classes)
    • Write per-class and combined dumps used by the evaluator
    """

    def __init__(self, cfg: Dict):
        # Map loose config dict → strongly-typed dataclass (with sensible defaults)
        self.cfg = MAFPipelineConfig(
            IMG_SHAPE=tuple(cfg.get("IMG_SHAPE", (40, 40, 1))),
            NUM_CLASSES=int(cfg.get("NUM_CLASSES", 9)),
            NUM_FLOWS=int(cfg.get("model", {}).get("num_flow_layers", cfg.get("NUM_FLOWS", 5))),
            HIDDEN_DIMS=tuple(cfg.get("model", {}).get("hidden_dims", cfg.get("HIDDEN_DIMS", (128, 128)))),
            EPOCHS=int(cfg.get("EPOCHS", 5)),
            BATCH_SIZE=int(cfg.get("BATCH_SIZE", 256)),
            LR=float(cfg.get("LR", 2e-4)),
            CLIP_GRAD=float(cfg.get("CLIP_GRAD", 1.0)),
            PATIENCE=int(cfg.get("PATIENCE", cfg.get("patience", 10))),
            SEED=int(cfg.get("SEED", 42)),
            SAMPLES_PER_CLASS=int(cfg.get("SAMPLES_PER_CLASS", 25)),
            SAMPLE_TEMPERATURE=float(cfg.get("SAMPLE_TEMPERATURE", 1.0)),
            ARTIFACTS=cfg.get("ARTIFACTS", {}),
        )

        arts = self.cfg.ARTIFACTS or {}
        self.ckpt_dir = Path(arts.get("checkpoints", "artifacts/maskedautoflow/checkpoints"))
        self.synth_dir = Path(arts.get("synthetic",   "artifacts/maskedautoflow/synthetic"))
        _ensure_dir(self.ckpt_dir)
        _ensure_dir(self.synth_dir)

        # Optional external logger callback: cb(stage: str, message: str)
        self.log_cb = cfg.get("LOG_CB", None)

        # Model handle + best checkpoint path
        self.model: Optional[MAF] = None
        self.best_ckpt_path: Optional[Path] = None

        # Reproducibility (seed once; avoid re-seeding inside hot paths)
        np.random.seed(self.cfg.SEED)
        tf.random.set_seed(self.cfg.SEED)

    # ----------------------- Logging -----------------------
    def _log(self, stage: str, msg: str) -> None:
        """Route logs to optional external callback or stdout."""
        if self.log_cb:
            try:
                self.log_cb(stage, msg)
                return
            except Exception:
                pass  # Fall back to stdout if callback misbehaves.
        print(f"[{stage}] {msg}")

    # ----------------------- Data utils --------------------
    def _make_dataset(self, x_flat: Optional[np.ndarray], batch: int, shuffle: bool) -> Optional[tf.data.Dataset]:
        """
        Wrap (N, D) float32 arrays into a simple tf.data pipeline.
        Returns None if x_flat is None (useful for optional val split).
        """
        if x_flat is None:
            return None
        x = np.asarray(x_flat, dtype=np.float32, order="C")
        ds = tf.data.Dataset.from_tensor_slices((x,))
        if shuffle:
            ds = ds.shuffle(
                buffer_size=min(len(x), 10000),
                seed=self.cfg.SEED,
                reshuffle_each_iteration=True,
            )
        return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

    # ----------------------- Training -----------------------
    def train(
        self,
        x_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,  # unused (unconditional)
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,    # unused (unconditional)
        *,
        save_checkpoints: bool = True,
    ) -> Dict:
        """
        Fit a global MAF on flattened images in [0,1].

        Parameters
        ----------
        x_train : np.ndarray
            Training images, (N,H,W,C) or (N,D) in [0,1].
        x_val : np.ndarray | None
            Optional validation images for early stopping.
        save_checkpoints : bool
            If True, write best/last weights to ckpt dir.

        Returns
        -------
        dict
            Minimal “bundle” with training metadata:
            {"input_dim": D, "best": <path-or-None>}
        """
        H, W, C = self.cfg.IMG_SHAPE
        D = H * W * C

        # Ensure (N, D) float32 in [0,1]
        Xtr = flatten_images(x_train, img_shape=(H, W, C), assume_01=True, clip=True)
        Xva = flatten_images(x_val,   img_shape=(H, W, C), assume_01=True, clip=True) if x_val is not None else None

        train_ds = self._make_dataset(Xtr, batch=self.cfg.BATCH_SIZE, shuffle=True)
        val_ds   = self._make_dataset(Xva, batch=self.cfg.BATCH_SIZE, shuffle=False)

        # Build model from config (Keras variables are created on first call)
        self.model = build_maf_model(
            MAFConfig(
                IMG_SHAPE=(H, W, C),
                NUM_FLOWS=self.cfg.NUM_FLOWS,
                HIDDEN_DIMS=self.cfg.HIDDEN_DIMS,
                LR=self.cfg.LR,
                CLIP_GRAD=self.cfg.CLIP_GRAD,
                PATIENCE=self.cfg.PATIENCE,
                RANDOM_STATE=self.cfg.SEED,
            )
        )
        _ = self.model(tf.zeros((1, D), dtype=tf.float32))  # warm build

        opt = tf.keras.optimizers.Adam(learning_rate=self.cfg.LR)
        best_val = np.inf
        bad_epochs = 0

        self._log(
            "train",
            f"Training MAF(D={D}, flows={self.cfg.NUM_FLOWS}, hidden={self.cfg.HIDDEN_DIMS}) "
            f"epochs={self.cfg.EPOCHS} bs={self.cfg.BATCH_SIZE} lr={self.cfg.LR}"
        )

        for epoch in range(1, self.cfg.EPOCHS + 1):
            # ---- Train epoch ----
            m_train = tf.keras.metrics.Mean()
            for (xb,) in train_ds:  # type: ignore[arg-type]
                with tf.GradientTape() as tape:
                    nll = -tf.reduce_mean(self.model.log_prob(xb))  # negative log-likelihood
                grads = tape.gradient(nll, self.model.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, self.cfg.CLIP_GRAD)
                opt.apply_gradients(zip(grads, self.model.trainable_variables))
                m_train.update_state(nll)

            # ---- Validate (if provided) ----
            val_nll = np.nan
            if val_ds is not None:
                m_val = tf.keras.metrics.Mean()
                for (xb,) in val_ds:  # type: ignore[arg-type]
                    m_val.update_state(-tf.reduce_mean(self.model.log_prob(xb)))
                val_nll = float(m_val.result().numpy())

            self._log(
                "train",
                f"epoch {epoch:03d}: train_nll={m_train.result().numpy():.4f}"
                + (f" | val_nll={val_nll:.4f}" if val_ds is not None else "")
            )

            # ---- Early stopping / checkpointing ----
            # If no val set, treat the final epoch as "best" for saving.
            improved = (val_ds is None and epoch == self.cfg.EPOCHS) or (val_ds is not None and val_nll < best_val)
            if improved:
                best_val = val_nll if val_ds is not None else best_val
                bad_epochs = 0
                if save_checkpoints:
                    best_path = self.ckpt_dir / "MAF_best.weights.h5"
                    self.model.save_weights(str(best_path), overwrite=True)  # Keras 3: no save_format
                    self.best_ckpt_path = best_path
                    self._log("ckpt", f"Saved {best_path.name}")
            else:
                bad_epochs += 1
                if val_ds is not None and bad_epochs >= self.cfg.PATIENCE:
                    self._log("train", "Early stopping triggered.")
                    break

        # Always save a "last" snapshot (useful for debugging/resume)
        if save_checkpoints:
            last_path = self.ckpt_dir / "MAF_last.weights.h5"
            self.model.save_weights(str(last_path), overwrite=True)  # Keras 3: no save_format
            (self.ckpt_dir / "MAF_LAST_OK").write_text("ok", encoding="utf-8")

        return {"input_dim": D, "best": str(self.best_ckpt_path) if self.best_ckpt_path else None}

    # Back-compat: some runners call fit()
    def fit(self, *args, **kwargs):
        """Alias for `train` to match existing orchestration code."""
        return self.train(*args, **kwargs)

    # ----------------------- Checkpoints --------------------
    def _load_best(self, D: int) -> MAF:
        """
        Lazy-build the model if needed and load best (or last) weights.
        """
        if self.model is None:
            self.model = build_maf_model(
                MAFConfig(
                    IMG_SHAPE=self.cfg.IMG_SHAPE,
                    NUM_FLOWS=self.cfg.NUM_FLOWS,
                    HIDDEN_DIMS=self.cfg.HIDDEN_DIMS,
                )
            )
            _ = self.model(tf.zeros((1, D), dtype=tf.float32))

        best = self.ckpt_dir / "MAF_best.weights.h5"
        last = self.ckpt_dir / "MAF_last.weights.h5"
        to_load = best if best.exists() else last
        if not to_load.exists():
            raise FileNotFoundError(f"No MAF checkpoint found in {self.ckpt_dir}")
        self.model.load_weights(str(to_load))
        self._log("ckpt", f"Loaded {to_load.name}")
        return self.model

    # ----------------------- Sampling -----------------------
    def _sample_z(self, n: int, D: int, temperature: float) -> tf.Tensor:
        """
        Draw base noise for inverse sampling.
        τ < 1.0 → crisper / lower-variance; τ > 1.0 → more diverse.
        """
        # Do not re-seed here; rely on global seed set in __init__.
        z = tf.random.normal(shape=(n, D), dtype=tf.float32)
        return z * float(temperature)

    def _sample(self, n_total: int) -> np.ndarray:
        """
        Sample `n_total` unconditional images in [0,1], shape (n_total, H, W, C).
        """
        H, W, C = self.cfg.IMG_SHAPE
        D = H * W * C
        model = self._load_best(D)
        z = self._sample_z(n_total, D, temperature=self.cfg.SAMPLE_TEMPERATURE)
        x_flat = model.inverse(z).numpy().astype(np.float32)
        x_flat = np.clip(x_flat, 0.0, 1.0)
        return reshape_to_images(x_flat, (H, W, C), clip=True)

    def _emit_per_class_files(self, x: np.ndarray, per_class: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evenly split unconditional samples per class and write evaluator files.

        Artifacts written to `self.synth_dir`:
          • gen_class_{k}.npy        -> float32 images in [0,1], (Nk, H, W, C)
          • labels_class_{k}.npy     -> int labels (k), shape (Nk,)
          • x_synth.npy, y_synth.npy -> concatenated convenience dumps
        """
        K = self.cfg.NUM_CLASSES
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []

        for k in range(K):
            start, end = k * per_class, (k + 1) * per_class
            xk = x[start:end]
            np.save(self.synth_dir / f"gen_class_{k}.npy", xk)
            np.save(self.synth_dir / f"labels_class_{k}.npy", np.full((len(xk),), k, dtype=np.int32))
            xs.append(xk)

            y1h = np.zeros((len(xk), K), dtype=np.float32)
            y1h[:, k] = 1.0
            ys.append(y1h)

        x_synth = np.concatenate(xs, axis=0).astype(np.float32)
        y_synth = np.concatenate(ys, axis=0).astype(np.float32)

        # Final combined dumps
        np.save(self.synth_dir / "x_synth.npy", x_synth)
        np.save(self.synth_dir / "y_synth.npy", y_synth)
        return x_synth, y_synth

    def synthesize(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a class-balanced synthetic set and write per-class/combined dumps.

        Returns
        -------
        x_synth : (N, H, W, C)
        y_synth : (N, K) one-hot
        """
        _ensure_dir(self.synth_dir)
        K = self.cfg.NUM_CLASSES
        per_class = int(self.cfg.SAMPLES_PER_CLASS)
        n_total = K * per_class

        x = self._sample(n_total)

        # Sanity: drop non-finite, if any (rare, but protects downstream metrics)
        finite = np.isfinite(x).all(axis=(1, 2, 3))
        if not finite.all():
            dropped = int((~finite).sum())
            self._log("warn", f"Dropping {dropped} non-finite synthetic samples.")
            x = x[finite]
            # Top-up to maintain exact per_class per class (best-effort)
            deficit = n_total - len(x)
            if deficit > 0:
                x_more = self._sample(deficit)
                x = np.concatenate([x, x_more], axis=0)

        x_s, y_s = self._emit_per_class_files(x[:n_total], per_class=per_class)
        self._log("synthesize", f"{x_s.shape[0]} samples ({per_class} per class) -> {self.synth_dir}")
        return x_s, y_s

    # ----------------------- Preview helper ----------------
    def _sample_batch(self, *_, n_per_class: int = 1, **__) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (x, y) with approximately `n_per_class * K` samples for quick visualization.

        Accepts **kwargs so callers that pass extra keys (e.g., `bundle=...`)
        won’t fail. Nothing is written to disk.
        """
        K = self.cfg.NUM_CLASSES
        n_total = max(1, n_per_class) * K
        x = self._sample(n_total)

        # Build dummy balanced labels (one-hot) to match preview grid contract
        ints = np.repeat(np.arange(K, dtype=np.int32), max(1, n_per_class))
        y = _one_hot(ints, K)
        return x[: len(ints)], y
