from typing import Tuple

import tensorflow as tf


class Discriminator(tf.keras.models.Model):
    def __init__(self, input_resolution: int, a_channels: int, b_channels: int,  filters: int):
        img_A = tf.keras.layers.Input(shape=(input_resolution, input_resolution, a_channels))
        img_B = tf.keras.layers.Input(shape=(input_resolution, input_resolution, b_channels))

        # Stack images
        combined_imgs = tf.keras.layers.Concatenate(axis=3)([img_A, img_B])

        d1 = self._block(combined_imgs, filters=filters, batch_norm=False)
        d2 = self._block(d1, filters=filters * 2)
        d3 = self._block(d2, filters=filters * 4)
        d4 = self._block(d3, filters=filters * 8)

        output_layer = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        super().__init__([img_A, img_B], output_layer)

    def _block(self, input, filters: int, kernel_size: int = 4, batch_norm: bool = True):
        d = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=2, padding='same')(input)
        if batch_norm:
            d = tf.keras.layers.BatchNormalization(momentum=0.8)(d)
        d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)

        return d
