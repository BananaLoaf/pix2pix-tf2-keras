from typing import List

import tensorflow as tf
import numpy as np


class UNet(tf.keras.models.Model):
    def __init__(self, resolution: int, input_channels: int, output_channels: int, filters: int, n_blocks: int):
        self.filters = filters
        self.n_blocks = n_blocks

        input_layer = tf.keras.layers.Input(shape=(resolution, resolution, input_channels))
        layers = self.encoder(input=input_layer)
        output_layer = self.decoder(layers, channels=output_channels)

        super().__init__(input_layer, output_layer)

    def encoder(self, input: tf.keras.layers.Input) -> List[tf.Tensor]:
        layers = []

        e = self._conv2d(input, filters=self.filters, batch_norm=False)
        layers.append(e)

        for pwr in [2, 4, 8]:
            e = self._conv2d(e, filters=self.filters * pwr)
            layers.append(e)

        while len(layers) < self.n_blocks:
            e = self._conv2d(e, filters=self.filters * 8)
            layers.append(e)

        return layers

    def decoder(self, layers: List[tf.Tensor], channels: int) -> tf.Tensor:
        layers = reversed(layers)

        e = next(layers)
        skip = next(layers)
        d = self._deconv2d(e, skip, filters=skip.shape[3])

        while True:
            try:
                skip = next(layers)
                d = self._deconv2d(d, skip, filters=skip.shape[3])
            except StopIteration:
                break

        d = tf.keras.layers.UpSampling2D(size=2)(d)
        d = tf.keras.layers.Conv2D(channels, kernel_size=4, strides=1, padding='same',
                                   activation=tf.keras.activations.tanh, name="gen_output")(d)
        return d

    @staticmethod
    def _conv2d(input, filters: int, kernel_size: int = 4, batch_norm: bool = True) -> tf.Tensor:
        x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=2, padding='same')(input)
        if batch_norm:
            x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        x = tf.keras.layers.LeakyReLU()(x)

        return x

    @staticmethod
    def _deconv2d(input, skip_input, filters: int, kernel_size: int = 4, dropout_rate: int = 0) -> tf.Tensor:
        x = tf.keras.layers.UpSampling2D(size=2)(input)
        x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
        if dropout_rate:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Concatenate()([x, skip_input])

        return x

    def generate_samples(self, real_As: tf.Tensor, real_Bs: tf.Tensor, n: int):
        """Must return RGB image if possible, else None"""
        fake_As = self.predict(real_Bs)

        rgb_img = np.hstack([real_Bs[0], real_As[0], fake_As[0]])
        for row in range(1, n):
            rgb_img = np.vstack([
                rgb_img,
                np.hstack([real_Bs[row], real_As[row], fake_As[row]])
            ])
        rgb_img = ((rgb_img + 1) * 127.5).astype(np.uint8)

        return rgb_img


class UNet64(UNet):
    def __init__(self, resolution: int, input_channels: int, output_channels: int, filters: int = 64):
        super().__init__(resolution=resolution,
                         input_channels=input_channels,
                         output_channels=output_channels,
                         filters=filters, n_blocks=6)


class UNet128(UNet):
    def __init__(self, resolution: int, input_channels: int, output_channels: int, filters: int = 64):
        super().__init__(resolution=resolution,
                         input_channels=input_channels,
                         output_channels=output_channels,
                         filters=filters, n_blocks=7)


class UNet256(UNet):
    def __init__(self, resolution: int, input_channels: int, output_channels: int, filters: int = 64):
        super().__init__(resolution=resolution,
                         input_channels=input_channels,
                         output_channels=output_channels,
                         filters=filters, n_blocks=8)
