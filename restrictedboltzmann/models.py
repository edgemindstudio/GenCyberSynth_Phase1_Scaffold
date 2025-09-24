# restrictedboltzmann/models.py
"""
TensorFlow/Keras Restricted Boltzmann Machine (RBM) utilities.

What you get
------------
- RBMConfig:  Dataclass of core hyperparameters.
- RBM:        Minimal TF2/Keras RBM (Bernoulli visible/hidden) with:
                * prop-up/prop-down
                * Gibbs sampling (CD-k)
                * free energy
                * two train modes:
                    - 'cd':    classical contrastive-divergence update (manual)
                    - 'mse':   reconstruction MSE with autograd
- build_rbm(): Convenience constructor (seeding + variable build).
- Image helpers: to_float01, binarize01, flatten_images, reshape_to_images.
- Sampling helper: sample_gibbs() to draw images from the model.
- BernoulliRBM: Thin wrapper exposing expected methods for samplers:
                sample_h_given_v, sample_v_given_h.

Conventions
-----------
- Images are channels-last (H, W, C), values in [0,1]; binarization is optional
  but typical for Bernoulli RBMs.
- All computations are float32 by default; CD updates are applied with
  `assign_add` (optimizer-free) to match the classic algorithm.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import tensorflow as tf

# =======================
# Config (optional export)
# =======================
@dataclass
class RBMConfig:
    visible_units: int
    hidden_units: int = 256
    cd_k: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    train_mode: str = "cd"  # {'cd','mse'}
    seed: Optional[int] = 42

# =======================
# Small helpers (optional)
# =======================
def to_float01(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32", copy=False)
    if x.max() > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)

def binarize01(x: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    return (x >= float(thresh)).astype("float32")

def flatten_images(x: np.ndarray, img_shape: Tuple[int,int,int], *, assume_01: bool = True) -> np.ndarray:
    H, W, C = img_shape
    x = x.reshape((-1, H, W, C))
    if not assume_01:
        x = to_float01(x)
    return x.reshape((-1, H * W * C)).astype("float32", copy=False)

def reshape_to_images(x_flat: np.ndarray, img_shape: Tuple[int,int,int]) -> np.ndarray:
    H, W, C = img_shape
    return x_flat.reshape((-1, H, W, C)).astype("float32", copy=False)

# ===========
# Core RBM
# ===========
class RBM(tf.keras.Model):
    """
    Bernoulli–Bernoulli RBM with propup/propdown, CD-k and MSE modes.
    Variables:
      W      : (V,H)
      h_bias : (H,)
      v_bias : (V,)
    """
    def __init__(self, visible_units: int, hidden_units: int = 256, name: str = "rbm"):
        super().__init__(name=name)
        self.visible_units = int(visible_units)
        self.hidden_units  = int(hidden_units)
        init = tf.keras.initializers.RandomNormal(stddev=0.01)
        self.W      = tf.Variable(init(shape=(self.visible_units, self.hidden_units)), name="W")
        self.h_bias = tf.Variable(tf.zeros([self.hidden_units]), name="h_bias")
        self.v_bias = tf.Variable(tf.zeros([self.visible_units]), name="v_bias")

    @tf.function(jit_compile=False)
    def _sigmoid(self, x: tf.Tensor) -> tf.Tensor:
        return tf.math.sigmoid(x)

    @tf.function(jit_compile=False)
    def _bernoulli_sample(self, probs: tf.Tensor) -> tf.Tensor:
        rnd = tf.random.uniform(tf.shape(probs), dtype=probs.dtype)
        return tf.cast(rnd < probs, probs.dtype)

    @tf.function(jit_compile=False)
    def propup(self, v: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        logits = tf.linalg.matmul(v, self.W) + self.h_bias
        probs  = self._sigmoid(logits)
        return logits, probs

    @tf.function(jit_compile=False)
    def propdown(self, h: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        logits = tf.linalg.matmul(h, tf.transpose(self.W)) + self.v_bias
        probs  = self._sigmoid(logits)
        return logits, probs

    @tf.function(jit_compile=False)
    def gibbs_k(self, v0: tf.Tensor, k: int = 1) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        v = v0
        _, h_prob = self.propup(v)
        for _ in tf.range(k):
            h = self._bernoulli_sample(h_prob)
            _, v_prob = self.propdown(h)
            v = self._bernoulli_sample(v_prob)
            _, h_prob = self.propup(v)
        return v, h_prob, v_prob

    # forward recon (used by 'mse' mode)
    @tf.function(jit_compile=False)
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        _, h_prob = self.propup(inputs)
        _, v_prob = self.propdown(h_prob)
        return v_prob

    @tf.function(jit_compile=False)
    def free_energy(self, v: tf.Tensor) -> tf.Tensor:
        vbias_term = tf.reduce_sum(v * self.v_bias, axis=1)
        hidden_lin = tf.linalg.matmul(v, self.W) + self.h_bias
        hidden_term = tf.reduce_sum(tf.math.softplus(hidden_lin), axis=1)
        return -(vbias_term + hidden_term)

    @tf.function(jit_compile=False)
    def train_step_cd(self, v0: tf.Tensor, *, k: int = 1, lr: float = 1e-3, weight_decay: float = 0.0) -> tf.Tensor:
        _, h0_prob = self.propup(v0)
        vk, hk_prob, v_prob = self.gibbs_k(v0, k=k)
        B = tf.cast(tf.shape(v0)[0], v0.dtype)
        pos = tf.linalg.matmul(tf.transpose(v0), h0_prob) / B
        neg = tf.linalg.matmul(tf.transpose(vk), hk_prob) / B
        dW  = pos - neg - weight_decay * self.W
        dvb = tf.reduce_mean(v0 - vk, axis=0)
        dhb = tf.reduce_mean(h0_prob - hk_prob, axis=0)
        self.W.assign_add(lr * dW)
        self.v_bias.assign_add(lr * dvb)
        self.h_bias.assign_add(lr * dhb)
        return tf.reduce_mean(tf.square(v0 - v_prob))

    @tf.function(jit_compile=False)
    def train_step_mse(self, v0: tf.Tensor, optimizer: tf.keras.optimizers.Optimizer) -> tf.Tensor:
        with tf.GradientTape() as tape:
            v_hat = self(v0, training=True)
            loss  = tf.reduce_mean(tf.square(v0 - v_hat))
        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

# ==========================
# Thin wrapper for samplers
# ==========================
class BernoulliRBM(RBM):
    """
    Accepts `visible_dim`/`hidden_dim` and exposes:
      - sample_h_given_v, sample_v_given_h
      - builds variables immediately so load_weights(...) works
    """
    def __init__(self, *, visible_dim: Optional[int] = None, hidden_dim: int = 256, name: str = "rbm", **kwargs) -> None:
        if visible_dim is None:
            if "visible_units" in kwargs:
                visible_dim = int(kwargs.pop("visible_units"))
            else:
                raise TypeError("BernoulliRBM requires `visible_dim=` integer.")
        if "hidden_units" in kwargs:
            hidden_dim = int(kwargs.pop("hidden_units"))
        super().__init__(visible_units=int(visible_dim), hidden_units=int(hidden_dim), name=name)
        _ = self(tf.zeros((1, int(visible_dim)), dtype=tf.float32))

    @tf.function(reduce_retracing=True)
    def sample_h_given_v(self, v: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        _, h_prob = self.propup(v)
        h_sample = tf.cast(tf.random.uniform(tf.shape(h_prob)) < h_prob, tf.float32)
        return h_sample, h_prob

    @tf.function(reduce_retracing=True)
    def sample_v_given_h(self, h: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        _, v_prob = self.propdown(h)
        v_sample = tf.cast(tf.random.uniform(tf.shape(v_prob)) < v_prob, tf.float32)
        return v_sample, v_prob

__all__ = [
    "RBMConfig",
    "RBM",
    "BernoulliRBM",
    "to_float01",
    "binarize01",
    "flatten_images",
    "reshape_to_images",
]
