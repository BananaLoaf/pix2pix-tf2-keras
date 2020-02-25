from typing import Tuple

import tensorflow as tf


class Discriminator(tf.keras.models.Model):
    def __init__(self, input_resolution: int, input_channels: int, filters: int):
        img_PHOTO = tf.keras.layers.Input(shape=(input_resolution, input_resolution, input_channels), name="photo")
        img_LABEL = tf.keras.layers.Input(shape=(input_resolution, input_resolution, input_channels), name="label")

        # Stack images
        combined_imgs = tf.keras.layers.Concatenate(axis=3)([img_PHOTO, img_LABEL])

        d1 = self._block(combined_imgs, filters=filters, batch_norm=False)
        d2 = self._block(d1, filters=filters * 2)
        d3 = self._block(d2, filters=filters * 4)
        d4 = self._block(d3, filters=filters * 8)

        output_layer = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        super().__init__([img_PHOTO, img_LABEL], output_layer)

    def __repr__(self):
        return self.__class__.__name__

    def _block(self, input, filters: int, kernel_size: int = 4, batch_norm: bool = True):
        d = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=2, padding='same')(input)
        if batch_norm:
            d = tf.keras.layers.BatchNormalization(momentum=0.8)(d)
        d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)

        return d
