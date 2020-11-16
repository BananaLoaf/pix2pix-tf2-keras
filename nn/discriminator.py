from typing import Tuple, Optional

import tensorflow as tf
import tensorflow_addons as tfa


class Discriminator(tf.keras.models.Model):
    def __init__(self, input_resolution: int, a_channels: int, b_channels: int, filters: int, n_blocks: int,
                 norm_layer: str):
        norm_layer = {
            "BatchNormalization": tf.keras.layers.BatchNormalization,
            "InstanceNormalization": tfa.layers.InstanceNormalization,
        }[norm_layer]
        bias = norm_layer == tfa.layers.InstanceNormalization

        img_A = tf.keras.layers.Input(shape=(input_resolution, input_resolution, a_channels))
        img_B = tf.keras.layers.Input(shape=(input_resolution, input_resolution, b_channels))

        # Stack images
        combined_imgs = tf.keras.layers.Concatenate(axis=3)([img_A, img_B])

        d = self._block(combined_imgs, filters=filters, kernel_size=4, strides=2, bias=True)
        for i in range(1, n_blocks):
            mult = min(2 ** i, 8)
            d = self._block(d, filters=filters * mult, kernel_size=4, strides=2, bias=bias, norm_layer=norm_layer)

        mult = min(2 ** n_blocks, 8)
        d = self._block(d, filters=filters * mult, kernel_size=4, strides=1, bias=bias, norm_layer=norm_layer)

        output_layer = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same', use_bias=True)(d)

        super().__init__([img_A, img_B], output_layer)

    @staticmethod
    def _block(x, filters: int, kernel_size: int, strides: int, bias: bool,
               norm_layer: Optional = None):
        x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=bias)(x)

        if norm_layer is not None:
            x = norm_layer(momentum=0.1, epsilon=1e-5)(x)

        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        return x
