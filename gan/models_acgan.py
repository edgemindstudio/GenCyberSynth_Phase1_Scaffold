# gan/models_acgan.py
"""
ACGAN-style model builders for Paper 2.

This file intentionally preserves the baseline conditional generator architecture
from gan.models, but replaces the discriminator/combined objective with an
auxiliary class-prediction head.

Purpose
-------
Paper 2 tests whether explicit class supervision repairs conditioning collapse.

Outputs
-------
- Generator: same architecture as baseline Conditional_Generator.
- Discriminator:
    1. validity output: real/fake sigmoid
    2. class_out output: class softmax
- Combined ACGAN:
    1. validity output: generator should fool discriminator
    2. class_out output: generated image should match requested class
"""

from __future__ import annotations

from typing import Dict, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

from gan.models import build_generator

Adam = tf.keras.optimizers.Adam


def build_discriminator_acgan(
    img_shape: Tuple[int, int, int] = (40, 40, 1),
    num_classes: int = 9,
) -> tf.keras.Model:
    """
    ACGAN discriminator.

    It keeps the same conditional image+label input style as the baseline
    discriminator, but adds an auxiliary class-prediction head.

    Inputs:
        x_in: image tensor
        y_in: one-hot requested/paired label

    Outputs:
        validity: real/fake probability
        class_out: predicted class distribution
    """
    H, W, C = img_shape

    x_in = layers.Input(shape=img_shape, name="x")
    y_in = layers.Input(shape=(num_classes,), name="y_onehot")

    # Same conditional label-map injection as baseline cDCGAN.
    y_map = layers.Dense(H * W, use_bias=False, name="disc_label_dense")(y_in)
    y_map = layers.Reshape((H, W, 1), name="disc_label_map")(y_map)

    xy = layers.Concatenate(axis=-1, name="disc_xy_concat")([x_in, y_map])

    x = layers.Conv2D(64, 4, strides=2, padding="same", name="disc_conv1")(xy)
    x = layers.LeakyReLU(negative_slope=0.2, name="disc_lrelu1")(x)
    x = layers.Dropout(0.3, name="disc_dropout1")(x)

    x = layers.Conv2D(128, 4, strides=2, padding="same", name="disc_conv2")(x)
    x = layers.BatchNormalization(name="disc_bn2")(x)
    x = layers.LeakyReLU(negative_slope=0.2, name="disc_lrelu2")(x)
    x = layers.Dropout(0.3, name="disc_dropout2")(x)

    features = layers.Flatten(name="disc_features")(x)

    validity = layers.Dense(1, activation="sigmoid", name="validity")(features)
    class_out = layers.Dense(num_classes, activation="softmax", name="class_out")(features)

    return models.Model([x_in, y_in], [validity, class_out], name="ACGAN_Discriminator")


def build_models_acgan(
    latent_dim: int,
    num_classes: int,
    img_shape: Tuple[int, int, int],
    lr: float = 2e-4,
    beta_1: float = 0.5,
    validity_loss_weight: float = 1.0,
    class_loss_weight: float = 1.0,
) -> Dict[str, tf.keras.Model]:
    """
    Build and compile ACGAN models.

    Returns:
        {
            "generator": G,
            "discriminator": D,
            "gan": ACGAN
        }
    """
    G = build_generator(
        latent_dim=latent_dim,
        num_classes=num_classes,
        img_shape=img_shape,
    )

    D = build_discriminator_acgan(
        img_shape=img_shape,
        num_classes=num_classes,
    )

    d_opt = Adam(learning_rate=lr, beta_1=beta_1)
    g_opt = Adam(learning_rate=lr, beta_1=beta_1)

    D.compile(
        optimizer=d_opt,
        loss=["binary_crossentropy", "categorical_crossentropy"],
        loss_weights=[validity_loss_weight, class_loss_weight],
        metrics=[["accuracy"], ["accuracy"]],
    )

    z_in = layers.Input(shape=(latent_dim,), name="z_in")
    y_in = layers.Input(shape=(num_classes,), name="y_in")

    fake = G([z_in, y_in])

    # Freeze D inside the combined graph.
    D.trainable = False
    validity, class_out = D([fake, y_in])

    ACGAN = models.Model([z_in, y_in], [validity, class_out], name="ACGAN_Combined")
    ACGAN.compile(
        optimizer=g_opt,
        loss=["binary_crossentropy", "categorical_crossentropy"],
        loss_weights=[validity_loss_weight, class_loss_weight],
    )

    return {
        "generator": G,
        "discriminator": D,
        "gan": ACGAN,
    }


__all__ = [
    "build_discriminator_acgan",
    "build_models_acgan",
]