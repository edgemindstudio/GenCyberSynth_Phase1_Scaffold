# maskedautoflow/models.py

# =============================================================================
# Masked Autoregressive Flow (MAF) components for image vectors in [0,1].
#
# Highlights
# ----------
# • MADE-style masked dense layers with masks stored as *non-trainable weights*.
#   This avoids TF autograph / graph scoping issues present when holding masks
#   as raw tensors created inside call() or build().
# • Reproducible masks: degrees are drawn with a NumPy RNG seeded via
#   MAFConfig.RANDOM_STATE (propagated through build_maf_model).
# • Per-flow *permutations* (and inverse permutations) improve expressivity,
#   as is standard in MAF/IAF stacks. These are also reproducible.
# • Keras 3 compatible. No deprecated `save_format` usage here.
#
# API surface
# -----------
# - MAFConfig: small dataclass holding flow settings.
# - MaskedDense: MADE mask-respecting dense layer.
# - MADE: masked MLP producing (mu, log_sigma) for one flow step.
# - MAF: stack of MADE blocks with forward/log_prob/inverse.
# - build_maf_model: convenience factory from config or dimension.
# - flatten_images / reshape_to_images: light helpers for shape handling.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union, Optional

import numpy as np
import tensorflow as tf


# =============================================================================
# Config
# =============================================================================

@dataclass
class MAFConfig:
    """
    Configuration for a Masked Autoregressive Flow.

    Notes
    -----
    - IMG_SHAPE is used only for helpers; flows operate on flattened vectors.
    - HIDDEN_DIMS controls each MADE block's MLP width.
    - RANDOM_STATE seeds both TF and the NumPy RNG used for degrees/permutations.
    """
    IMG_SHAPE: Tuple[int, int, int] = (40, 40, 1)
    NUM_FLOWS: int = 5
    HIDDEN_DIMS: Tuple[int, ...] = (128, 128)
    # Training-time knobs (the pipeline typically uses these)
    LR: float = 2e-4
    CLIP_GRAD: float = 1.0
    PATIENCE: int = 10
    RANDOM_STATE: Optional[int] = 42


# =============================================================================
# Small helpers (shape transforms)
# =============================================================================

def flatten_images(x: np.ndarray,
                   img_shape: Tuple[int, int, int],
                   assume_01: bool = True,
                   clip: bool = True) -> np.ndarray:
    """
    Convert (N,H,W,C) -> (N, D) float32 in [0,1].

    If `x` is already (N,D), return a typed/clipped view.
    """
    x = np.asarray(x)
    if x.ndim == 4:
        H, W, C = img_shape
        x = x.reshape((-1, H * W * C))
    x = x.astype(np.float32, copy=False)
    if not assume_01 and x.max() > 1.5:  # best-effort normalization if bytes
        x = x / 255.0
    if clip:
        x = np.clip(x, 0.0, 1.0)
    return x


def reshape_to_images(x_flat: np.ndarray,
                      img_shape: Tuple[int, int, int],
                      clip: bool = True) -> np.ndarray:
    """
    Convert (N, D) -> (N,H,W,C) float32 in [0,1].
    """
    H, W, C = img_shape
    x = np.asarray(x_flat, dtype=np.float32)
    x = x.reshape((-1, H, W, C))
    if clip:
        x = np.clip(x, 0.0, 1.0)
    return x


# =============================================================================
# Core masked layers / MADE / MAF
# =============================================================================

class MaskedDense(tf.keras.layers.Layer):
    """
    Dense layer with a fixed binary mask applied to the kernel to enforce
    autoregressive constraints (MADE-style).

    Implementation detail
    ---------------------
    The mask is constructed with **NumPy** in `build()` and stored as a **non-
    trainable weight** (tf.Variable with `trainable=False`). This prevents TF
    autograph scoping problems that occur when keeping the mask as a raw Tensor
    captured from other graph contexts.
    """

    def __init__(
        self,
        units: int,
        in_degrees: np.ndarray,
        out_degrees: np.ndarray,
        use_bias: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.units = int(units)
        # Stash degrees as NumPy arrays; they are small and reproducible.
        self.in_degrees_np = np.asarray(in_degrees, dtype=np.int32)
        self.out_degrees_np = np.asarray(out_degrees, dtype=np.int32)
        self.use_bias = use_bias

        # Will be created in build()
        self.kernel: tf.Variable
        self.bias: Optional[tf.Variable]
        self.mask: tf.Variable  # non-trainable

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        if in_dim != int(self.in_degrees_np.shape[0]):
            raise ValueError(
                f"MaskedDense expected input dim {self.in_degrees_np.shape[0]}, got {in_dim}"
            )

        # Trainable kernel/bias
        self.kernel = self.add_weight(
            name="kernel",
            shape=(in_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        ) if self.use_bias else None

        # Fixed mask: 1 if in_degree <= out_degree
        mask_np = (self.in_degrees_np[:, None] <= self.out_degrees_np[None, :]).astype("float32")
        self.mask = self.add_weight(
            name="mask",
            shape=mask_np.shape,
            dtype=tf.float32,
            initializer=tf.constant_initializer(mask_np),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        masked_kernel = tf.multiply(self.kernel, self.mask)
        y = tf.linalg.matmul(inputs, masked_kernel)
        if self.bias is not None:
            y = tf.nn.bias_add(y, self.bias)
        return y


class MADE(tf.keras.Model):
    """
    MADE network: masked MLP producing (mu, log_sigma).

    `seed` ensures the degree assignments are reproducible.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: Sequence[int] = (128, 128),
                 seed: Optional[int] = None,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)

        # Reproducible degrees via a seeded NumPy RNG
        rng = np.random.default_rng(seed)
        deg_in = np.arange(1, self.input_dim + 1, dtype=np.int32)
        deg_hidden = [
            rng.integers(1, self.input_dim + 1, size=h, dtype=np.int32)
            for h in self.hidden_dims
        ]
        deg_out = np.arange(1, self.input_dim + 1, dtype=np.int32)

        # Build masked MLP
        layers_seq: List[tf.keras.layers.Layer] = []
        prev_deg = deg_in
        for li, h in enumerate(self.hidden_dims):
            layers_seq.append(MaskedDense(h, in_degrees=prev_deg, out_degrees=deg_hidden[li], name=f"mdense_{li}"))
            layers_seq.append(tf.keras.layers.ReLU())
            prev_deg = deg_hidden[li]

        self.net = layers_seq
        self.mu_layer = MaskedDense(self.input_dim, in_degrees=prev_deg, out_degrees=deg_out, name="mu")
        self.log_sigma_layer = MaskedDense(self.input_dim, in_degrees=prev_deg, out_degrees=deg_out, name="log_sigma")

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        h = x
        for layer in self.net:
            h = layer(h)
        mu = self.mu_layer(h)
        log_sigma = self.log_sigma_layer(h)
        # Clamp for numerical stability
        log_sigma = tf.clip_by_value(log_sigma, -7.0, 7.0)
        return mu, log_sigma


class MAF(tf.keras.Model):
    """
    Masked Autoregressive Flow (stack of MADE transforms).

    Forward (density):      x -> (permute) -> MADE -> z
    Inverse (sampling):     z -> MADE^{-1} -> (inverse permute) -> x

    The per-flow permutations increase expressivity. Both permutations and
    degrees are reproducible with `random_state`.
    """

    def __init__(self,
                 input_dim: int,
                 num_flows: int = 5,
                 hidden_dims: Sequence[int] = (128, 128),
                 random_state: Optional[int] = None,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.input_dim = int(input_dim)
        self.num_flows = int(num_flows)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)

        base = 0 if random_state is None else int(random_state)
        rng = np.random.default_rng(base)

        # ----- Deterministic permutations per flow -----
        self._perms_np: List[np.ndarray] = [rng.permutation(self.input_dim).astype(np.int32)
                                            for _ in range(self.num_flows)]
        self._inv_perms_np: List[np.ndarray] = [np.argsort(p).astype(np.int32) for p in self._perms_np]

        # Created in build() as TF constants (so they live in the right graph)
        self._perms_tf: List[tf.Tensor] = []
        self._inv_perms_tf: List[tf.Tensor] = []

        # ----- MADE blocks (each with a distinct, reproducible seed) -----
        self.flows: List[MADE] = [
            MADE(self.input_dim, self.hidden_dims, seed=base + i, name=f"made_{i}")
            for i in range(self.num_flows)
        ]

    def build(self, input_shape):
        # Materialize permutation tensors once in the current graph
        self._perms_tf = [tf.constant(p, dtype=tf.int32) for p in self._perms_np]
        self._inv_perms_tf = [tf.constant(p, dtype=tf.int32) for p in self._inv_perms_np]

        # Warm-build variables by calling once
        dummy = tf.zeros((1, self.input_dim), dtype=tf.float32)
        _ = self.call(dummy)
        super().build(input_shape)

    # ----- Forward transform and log-det -----
    def forward(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        x -> z, returns (z, log_det_jacobian).
        """
        z = x
        log_det = tf.zeros((tf.shape(x)[0],), dtype=tf.float32)
        for i, flow in enumerate(self.flows):
            # Permute *before* flow
            z = tf.gather(z, self._perms_tf[i], axis=1)
            mu, log_sigma = flow(z)
            z = (z - mu) * tf.exp(-log_sigma)
            log_det += -tf.reduce_sum(log_sigma, axis=1)
        return z, log_det

    # Keras `call` returns only z to remain a valid layer/model
    def call(self, x: tf.Tensor) -> tf.Tensor:
        z, _ = self.forward(x)
        return z

    # ----- Log-likelihood under standard normal base -----
    def log_prob(self, x: tf.Tensor) -> tf.Tensor:
        z, log_det = self.forward(x)
        # Diagonal standard normal base
        log_base = -0.5 * tf.reduce_sum(z * z + tf.math.log(2.0 * np.pi), axis=1)
        return log_base + log_det

    # ----- Inverse of a single flow (sequential across dimensions) -----
    @staticmethod
    def _inverse_single_flow(z: tf.Tensor, flow: MADE, input_dim: int) -> tf.Tensor:
        """
        Given z and one MADE flow, recover x such that:
            z = (x - mu(x)) * exp(-log_sigma(x))
        We solve per-dimension using the MADE masks: x_i depends only on x_{<i}.
        """
        batch = tf.shape(z)[0]
        D = int(input_dim)  # known statically for our usage
        x = tf.zeros_like(z)
        for i in range(D):
            mu, log_sigma = flow(x)  # valid: mu_i, sigma_i depend on x_<i
            xi = mu[:, i] + tf.exp(log_sigma[:, i]) * z[:, i]
            # assign xi -> x[:, i]
            xi = tf.reshape(xi, (batch, 1))
            left = x[:, :i]
            right = x[:, i + 1:]
            x = tf.concat([left, xi, right], axis=1)
        return x

    # ----- Full inverse across stacked flows (apply in reverse order) -----
    def inverse(self, z: tf.Tensor) -> tf.Tensor:
        """
        Invert the sequence:
            z_{i+1} = f_i( P_i( z_i ) )
        => z_i = P_i^{-1}( f_i^{-1}( z_{i+1} ) )
        Apply flows in reverse, first invert the flow, then invert the permutation.
        """
        x = z
        for i, flow in reversed(list(enumerate(self.flows))):
            # 1) invert MADE
            x = self._inverse_single_flow(x, flow, self.input_dim)
            # 2) invert permutation
            x = tf.gather(x, self._inv_perms_tf[i], axis=1)
        return x


# =============================================================================
# Factory
# =============================================================================

def build_maf_model(cfg_or_dim: Union[int, MAFConfig],
                    num_layers: Optional[int] = None,
                    hidden_dims: Optional[Iterable[int]] = None) -> MAF:
    """
    Build a MAF instance either from a config or from bare dimensions.

    Examples
    --------
    >>> model = build_maf_model(1600, num_layers=5, hidden_dims=(128,128))
    >>> model = build_maf_model(MAFConfig(IMG_SHAPE=(40,40,1), NUM_FLOWS=6))
    """
    if isinstance(cfg_or_dim, MAFConfig):
        D = int(np.prod(cfg_or_dim.IMG_SHAPE))
        flows = cfg_or_dim.NUM_FLOWS
        h = tuple(cfg_or_dim.HIDDEN_DIMS)
        maf = MAF(
            input_dim=D,
            num_flows=flows,
            hidden_dims=h,
            random_state=cfg_or_dim.RANDOM_STATE,
        )
    else:
        D = int(cfg_or_dim)
        flows = int(num_layers) if num_layers is not None else 5
        h = tuple(hidden_dims) if hidden_dims is not None else (128, 128)
        maf = MAF(input_dim=D, num_flows=flows, hidden_dims=h)

    # Eager-build variables/permutation tensors for safety
    maf.build((None, D))
    return maf
