# gan/train_acgan.py
"""
Train an ACGAN-style conditional GAN for Paper 2.

This trainer is intentionally separate from gan/train.py so the baseline cDCGAN
remains untouched and reproducible.

Key difference from baseline:
    The discriminator has two outputs:
        1. validity: real/fake
        2. class_out: predicted class

Generator objective:
    1. fool discriminator as real
    2. be classified as the requested class

Checkpoint compatibility:
    The generator architecture is reused from gan.models.build_generator, so
    existing gan/sample.py can synthesize from ACGAN generator checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from common.data import load_dataset_npy, to_minus1_1
from gan.models_acgan import build_models_acgan

try:
    from eval_common import fid_keras as compute_fid_01  # type: ignore
except Exception:
    compute_fid_01 = None


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now_ts()}] {msg}")


def _set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _enable_gpu_mem_growth() -> None:
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass


def _artifacts_root(cfg: dict) -> Path:
    return Path(cfg.get("paths", {}).get("artifacts", "artifacts"))


def _ensure_dirs(arts_root: Path, seed: int, config_id: str = "default") -> dict[str, Path]:
    config_id = str(config_id or "default")
    paths = {
        "ckpts": arts_root / "gan" / "checkpoints" / config_id / f"seed{seed}",
        "synthetic": arts_root / "gan" / "synthetic" / config_id / f"seed{seed}",
        "summaries": arts_root / "gan" / "summaries" / config_id / f"seed{seed}",
        "tensorboard": arts_root / "gan" / "tensorboard" / config_id / f"seed{seed}",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _to_float(x) -> float:
    if isinstance(x, (list, tuple)):
        x = x[0]
    try:
        return float(x)
    except Exception:
        try:
            return float(np.asarray(x).reshape(-1)[0])
        except Exception:
            return float(np.array(x).reshape(()).item())


def _loss_total(x) -> float:
    """
    Keras multi-output train_on_batch returns a list where index 0 is total loss.
    """
    return _to_float(x[0] if isinstance(x, (list, tuple)) else x)


def _maybe_add_noise(x: np.ndarray, std: float) -> np.ndarray:
    if std <= 0:
        return x
    return x + np.random.normal(0.0, std, size=x.shape).astype(np.float32)


def _save_grid(
    images01: np.ndarray,
    img_shape: Tuple[int, int, int],
    rows: int,
    cols: int,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    H, W, C = img_shape
    n = min(rows * cols, images01.shape[0])

    plt.figure(figsize=(cols * 1.6, rows * 1.6))
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        im = images01[i].reshape(H, W, C)
        if C == 1:
            plt.imshow(im.squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
        else:
            plt.imshow(np.clip(im, 0.0, 1.0))
        ax.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_from_file(
    cfg_path: Path,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    eval_every: int = 25,
    save_every: int = 50,
    label_smooth: tuple[float, float] = (0.9, 1.0),
    fake_label_range: tuple[float, float] = (0.0, 0.1),
    noise_after: int = 200,
    noise_std: float = 0.01,
    grid: tuple[int, int] | None = None,
    g_weights: Path | None = None,
    d_weights: Path | None = None,
    sample_after: bool = False,
    samples_per_class: int = 0,
    seed: int | None = None,
) -> int:
    _enable_gpu_mem_growth()

    import yaml

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    seed = int(seed if seed is not None else cfg.get("SEED", 42))
    _set_seeds(seed)

    train_cfg = cfg.get("train", {}) or {}
    gan_cfg = cfg.get("gan", {}) or {}
    acgan_cfg = cfg.get("acgan", {}) or {}

    IMG_SHAPE = tuple(cfg.get("IMG_SHAPE", train_cfg.get("IMG_SHAPE", (40, 40, 1))))
    NUM_CLASSES = int(cfg.get("NUM_CLASSES", train_cfg.get("NUM_CLASSES", 9)))
    LATENT_DIM = int(gan_cfg.get("latent_dim", cfg.get("LATENT_DIM", 100)))

    EPOCHS = int(
        epochs if epochs is not None else
        train_cfg.get("epochs", cfg.get("EPOCHS", 5000))
    )

    BATCH_SIZE = int(
        batch_size if batch_size is not None else
        train_cfg.get("batch_size", cfg.get("BATCH_SIZE", 256))
    )

    LR = float(gan_cfg.get("lr", cfg.get("LR", 2e-4)))
    BETA_1 = float(gan_cfg.get("beta_1", cfg.get("BETA_1", 0.5)))

    VALIDITY_W = float(acgan_cfg.get("validity_loss_weight", 1.0))
    CLASS_W = float(acgan_cfg.get("class_loss_weight", 1.0))

    DATA_DIR = Path(
        cfg.get(
            "DATA_DIR",
            cfg.get("DATA_PATH", Path(cfg_path).resolve().parents[1] / "USTC-TFC2016_malware"),
        )
    )

    TUNING_LITE = bool(cfg.get("TUNING_LITE", False))
    if TUNING_LITE and EPOCHS > 200:
        raise ValueError(
            f"TUNING_LITE is true but epochs={EPOCHS} (>200). "
            "Refusing to run."
        )

    rm = cfg.get("run_meta") if isinstance(cfg.get("run_meta"), dict) else {}
    config_id = (
        rm.get("config_id")
        or cfg.get("config_id")
        or cfg.get("model_variant")
        or "default"
    )

    arts_root = _artifacts_root(cfg)
    paths = _ensure_dirs(arts_root, seed, config_id)

    tb_run_dir = paths["tensorboard"] / f"acgan_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = tf.summary.create_file_writer(str(tb_run_dir))

    _log(
        f"[ACGAN] Config: HWC={IMG_SHAPE}, K={NUM_CLASSES}, Z={LATENT_DIM}, "
        f"epochs={EPOCHS}, bs={BATCH_SIZE}, lr={LR}, beta1={BETA_1}, "
        f"validity_w={VALIDITY_W}, class_w={CLASS_W}"
    )
    _log(f"[ACGAN] DATA_DIR={DATA_DIR}")
    _log(f"[ACGAN] TensorBoard → {tb_run_dir}")

    x_train01, y_train_oh, x_val01, y_val_oh, x_test01, y_test_oh = load_dataset_npy(
        DATA_DIR,
        IMG_SHAPE,
        NUM_CLASSES,
        val_fraction=float(cfg.get("VAL_FRACTION", 0.5)),
    )

    x_train = to_minus1_1(x_train01)

    nets = build_models_acgan(
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        img_shape=IMG_SHAPE,
        lr=LR,
        beta_1=BETA_1,
        validity_loss_weight=VALIDITY_W,
        class_loss_weight=CLASS_W,
    )

    G: tf.keras.Model = nets["generator"]
    D: tf.keras.Model = nets["discriminator"]
    ACGAN: tf.keras.Model = nets["gan"]

    if g_weights and Path(g_weights).exists():
        G.load_weights(str(g_weights))
        _log(f"[ACGAN] Loaded generator weights from {g_weights}")

    if d_weights and Path(d_weights).exists():
        D.load_weights(str(d_weights))
        _log(f"[ACGAN] Loaded discriminator weights from {d_weights}")

    steps_per_epoch = math.ceil(x_train.shape[0] / BATCH_SIZE)
    best_fid = float("inf")

    _log(f"[ACGAN] Start training for {EPOCHS} epochs ({steps_per_epoch} steps/epoch).")

    for epoch in range(1, EPOCHS + 1):
        idx = np.random.permutation(x_train.shape[0])

        d_losses = []
        g_losses = []
        d_class_losses = []
        g_class_losses = []

        for step in range(steps_per_epoch):
            batch_idx = idx[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
            real_imgs = x_train[batch_idx]
            real_lbls = y_train_oh[batch_idx]
            n = real_imgs.shape[0]

            if epoch > noise_after and noise_std > 0:
                real_imgs = _maybe_add_noise(real_imgs, noise_std)

            z = np.random.normal(0.0, 1.0, size=(n, LATENT_DIM)).astype(np.float32)
            fake_class_int = np.random.randint(0, NUM_CLASSES, size=(n,))
            fake_lbls = tf.keras.utils.to_categorical(fake_class_int, NUM_CLASSES).astype(np.float32)
            gen_imgs = G.predict([z, fake_lbls], verbose=0)

            if epoch > noise_after and noise_std > 0:
                gen_imgs = _maybe_add_noise(gen_imgs, noise_std)

            real_y = np.random.uniform(label_smooth[0], label_smooth[1], size=(n, 1)).astype(np.float32)
            fake_y = np.random.uniform(fake_label_range[0], fake_label_range[1], size=(n, 1)).astype(np.float32)

            D.trainable = True

            d_loss_real = D.train_on_batch(
                [real_imgs, real_lbls],
                [real_y, real_lbls],
            )

            # For fake images, train the validity head as fake.
            # The class head receives zero sample weight here so it is not
            # trained to trust early low-quality fake-image class labels.
            d_loss_fake = D.train_on_batch(
                [gen_imgs, fake_lbls],
                [fake_y, fake_lbls],
                sample_weight=[
                    np.ones((n,), dtype=np.float32),
                    np.zeros((n,), dtype=np.float32),
                ],
            )

            d_loss = 0.5 * (_loss_total(d_loss_real) + _loss_total(d_loss_fake))
            d_losses.append(d_loss)

            if isinstance(d_loss_real, (list, tuple)) and len(d_loss_real) >= 3:
                d_class_losses.append(float(d_loss_real[2]))

            D.trainable = False

            z = np.random.normal(0.0, 1.0, size=(BATCH_SIZE, LATENT_DIM)).astype(np.float32)
            g_lbls_int = np.random.randint(0, NUM_CLASSES, size=(BATCH_SIZE,))
            g_lbls = tf.keras.utils.to_categorical(g_lbls_int, NUM_CLASSES).astype(np.float32)

            g_loss_out = ACGAN.train_on_batch(
                [z, g_lbls],
                [
                    np.ones((BATCH_SIZE, 1), dtype=np.float32),
                    g_lbls,
                ],
            )

            g_loss = _loss_total(g_loss_out)
            g_losses.append(g_loss)

            if isinstance(g_loss_out, (list, tuple)) and len(g_loss_out) >= 3:
                g_class_losses.append(float(g_loss_out[2]))

        d_loss_ep = float(np.mean(d_losses))
        g_loss_ep = float(np.mean(g_losses))
        d_cls_ep = float(np.mean(d_class_losses)) if d_class_losses else float("nan")
        g_cls_ep = float(np.mean(g_class_losses)) if g_class_losses else float("nan")

        if grid is not None:
            rows, cols = grid
            z = np.random.normal(0.0, 1.0, size=(rows * cols, LATENT_DIM)).astype(np.float32)
            cyc = np.arange(rows * cols) % NUM_CLASSES
            y_cyc = tf.keras.utils.to_categorical(cyc, NUM_CLASSES).astype(np.float32)
            g = G.predict([z, y_cyc], verbose=0)
            g01 = np.clip((g + 1.0) / 2.0, 0.0, 1.0)
            _save_grid(g01, IMG_SHAPE, rows, cols, paths["summaries"] / f"acgan_grid_epoch_{epoch:04d}.png")

        fid_val = None
        if (compute_fid_01 is not None) and (epoch % max(1, eval_every) == 0):
            n_fid = min(200, x_val01.shape[0])
            real01 = x_val01[:n_fid]

            z = np.random.normal(0.0, 1.0, size=(n_fid, LATENT_DIM)).astype(np.float32)
            labels_int = np.random.randint(0, NUM_CLASSES, size=(n_fid,))
            y_oh = tf.keras.utils.to_categorical(labels_int, NUM_CLASSES).astype(np.float32)

            g = G.predict([z, y_oh], verbose=0)
            fake01 = np.clip((g + 1.0) / 2.0, 0.0, 1.0)

            try:
                fid_val = float(compute_fid_01(real01, fake01))  # type: ignore
            except Exception:
                fid_val = None

            if fid_val is not None and fid_val < best_fid:
                best_fid = fid_val
                G.save_weights(str(paths["ckpts"] / "G_best.weights.h5"))
                D.save_weights(str(paths["ckpts"] / "D_best.weights.h5"))
                with open(paths["ckpts"] / "best_fid.json", "w") as f:
                    json.dump(
                        {
                            "epoch": epoch,
                            "best_fid": best_fid,
                            "timestamp": _now_ts(),
                            "variant": "acgan_auxloss",
                        },
                        f,
                    )
                _log(f"[ACGAN][BEST] Epoch {epoch} new best FID={best_fid:.4f}")

        if (epoch % max(1, save_every) == 0) or (epoch == EPOCHS):
            G.save_weights(str(paths["ckpts"] / "G_last.weights.h5"))
            D.save_weights(str(paths["ckpts"] / "D_last.weights.h5"))
            G.save_weights(str(paths["ckpts"] / f"G_epoch_{epoch:04d}.weights.h5"))
            D.save_weights(str(paths["ckpts"] / f"D_epoch_{epoch:04d}.weights.h5"))

        msg = (
            f"[ACGAN] Epoch {epoch:04d} | "
            f"D_loss {d_loss_ep:.4f} | G_loss {g_loss_ep:.4f} | "
            f"D_cls {d_cls_ep:.4f} | G_cls {g_cls_ep:.4f}"
        )
        if fid_val is not None:
            msg += f" | FID {fid_val:.4f}"
        _log(msg)

        with writer.as_default():
            tf.summary.scalar("loss/D_total", d_loss_ep, step=epoch)
            tf.summary.scalar("loss/G_total", g_loss_ep, step=epoch)
            tf.summary.scalar("loss/D_class_real", d_cls_ep, step=epoch)
            tf.summary.scalar("loss/G_class", g_cls_ep, step=epoch)
            if fid_val is not None:
                tf.summary.scalar("FID/val", fid_val, step=epoch)
        writer.flush()

    _log("[ACGAN] Training complete.")

    if sample_after and samples_per_class > 0:
        out_dir = paths["synthetic"] / f"post_train_acgan_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        _log(f"[ACGAN] Sampling {samples_per_class} per class → {out_dir}")

        for k in range(NUM_CLASSES):
            z = np.random.normal(0.0, 1.0, size=(samples_per_class, LATENT_DIM)).astype(np.float32)
            y = tf.keras.utils.to_categorical(np.full((samples_per_class,), k), NUM_CLASSES).astype(np.float32)
            g = G.predict([z, y], verbose=0)
            g01 = np.clip((g + 1.0) / 2.0, 0.0, 1.0)
            np.save(out_dir / f"gen_class_{k}.npy", g01)
            np.save(out_dir / f"labels_class_{k}.npy", np.full((samples_per_class,), k, dtype=np.int32))

        _log("[ACGAN] Sampling done.")

    return 0


def train(cfg_or_argv):
    if isinstance(cfg_or_argv, (list, tuple)):
        return main(cfg_or_argv)

    if isinstance(cfg_or_argv, dict):
        import tempfile
        import yaml

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(cfg_or_argv, f)
            tmp = f.name

        return main(["--config", tmp])

    raise TypeError(f"Unsupported config payload type: {type(cfg_or_argv)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train ACGAN auxiliary-loss variant")

    default_cfg = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
    parser.add_argument("--config", type=Path, default=default_cfg, help="Path to config.yaml")

    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--eval-every", type=int, default=25, help="Evaluate FID every N epochs")
    parser.add_argument("--save-every", type=int, default=50, help="Save checkpoints every N epochs")

    parser.add_argument(
        "--label-smooth",
        type=float,
        nargs=2,
        default=(0.9, 1.0),
        help="Real label smoothing range [low high]",
    )
    parser.add_argument(
        "--fake-label-range",
        type=float,
        nargs=2,
        default=(0.0, 0.1),
        help="Fake label range [low high]",
    )

    parser.add_argument("--noise-after", type=int, default=200, help="Start noise injection after this epoch")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Gaussian noise std after warmup")

    parser.add_argument("--grid", type=int, nargs=2, default=None, help="Save preview grid: ROWS COLS per epoch")

    parser.add_argument("--g-weights", type=Path, default=None, help="Resume generator weights")
    parser.add_argument("--d-weights", type=Path, default=None, help="Resume discriminator weights")

    parser.add_argument("--sample-after", action="store_true", help="Generate per-class samples after training")
    parser.add_argument("--samples-per-class", type=int, default=0, help="Samples per class if --sample-after")

    parser.add_argument("--seed", type=int, default=None, help="Random seed override")

    args = parser.parse_args(argv)

    return run_from_file(
        cfg_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        save_every=args.save_every,
        label_smooth=tuple(args.label_smooth),
        fake_label_range=tuple(args.fake_label_range),
        noise_after=args.noise_after,
        noise_std=args.noise_std,
        grid=tuple(args.grid) if args.grid else None,
        g_weights=args.g_weights,
        d_weights=args.d_weights,
        sample_after=args.sample_after,
        samples_per_class=int(args.samples_per_class),
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())