from typing import List, Callable, Optional

import tensorflow as tf
import tensorflow_addons as tfa


class UNet(tf.keras.models.Model):
    def __init__(self, resolution: int, input_channels: int, output_channels: int, filters: int, n_blocks: int,
                 norm_layer: str, dropout: bool):
        self.filters = filters
        self.n_blocks = n_blocks

        self.norm_layer = {
            "BatchNormalization": tf.keras.layers.BatchNormalization,
            "InstanceNormalization": tfa.layers.InstanceNormalization,
        }[norm_layer]
        self.bias = self.norm_layer == tfa.layers.InstanceNormalization
        self.dropout = dropout

        # input_layer = tf.keras.layers.Input(shape=(resolution, resolution, input_channels))
        # norm = Switch(run=tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1))(input_layer)
        #
        # layers = self.encoder(input=norm)
        # decoder_layer = self.decoder(layers, channels=output_channels)
        #
        # output_layer = Switch(run=tf.keras.layers.experimental.preprocessing.Rescaling(127.5, offset=127.5))(decoder_layer)
        #
        # super().__init__(input_layer, output_layer)

        input_layer = tf.keras.layers.Input(shape=(resolution, resolution, input_channels))
        layers = self.encoder(input=input_layer)
        decoder_layer = self.decoder(layers, channels=output_channels)

        super().__init__(input_layer, decoder_layer)

    def encoder(self, input: tf.keras.layers.Input) -> List[tf.Tensor]:
        layers = []

        # First
        e = self._conv2d(input, filters=self.filters, kernel_size=4, strides=2, bias=self.bias, relu=False,
                         norm_layer=None)
        layers.append(e)

        # Middle
        for i in range(1, self.n_blocks - 1):
            mult = min(2 ** i, 8)
            e = self._conv2d(e, filters=self.filters * mult, kernel_size=4, strides=2, bias=self.bias, relu=True,
                             norm_layer=self.norm_layer)
            layers.append(e)

        # Last
        mult = min(2 ** self.n_blocks, 8)
        e = self._conv2d(e, filters=self.filters * mult, kernel_size=4, strides=2, bias=self.bias, relu=True,
                         norm_layer=None)
        layers.append(e)

        return layers

    def decoder(self, layers: List[tf.Tensor], channels: int) -> tf.Tensor:
        layers = list(reversed(layers))

        d = layers[0]
        for i, skip in enumerate(layers[1:]):
            filters = skip.shape[3]

            try:
                # If amount of filters is the same as in the next layer
                dropout = self.dropout and filters == layers[1:][i+1].shape[3]
            except IndexError:
                dropout = False

            d = self._deconv2d(d, skip, filters=filters, kernel_size=4, strides=2, bias=self.bias, dropout=dropout,
                               norm_layer=self.norm_layer)

        d = tf.keras.layers.ReLU()(d)
        d = tf.keras.layers.Conv2DTranspose(channels, kernel_size=4, strides=2, padding='same', use_bias=True,
                                            activation=tf.keras.activations.tanh)(d)
        return d

    @staticmethod
    def _conv2d(x, filters: int, kernel_size: int, strides: int, bias: bool,
                relu: bool, norm_layer: Optional) -> tf.Tensor:
        if relu:
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=bias)(x)

        if norm_layer is not None:
            x = norm_layer(momentum=0.1, epsilon=1e-5)(x)

        return x

    @staticmethod
    def _deconv2d(x, skip_x, filters: int, kernel_size: int, strides: int, bias: bool, dropout: bool, norm_layer) -> tf.Tensor:
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, use_bias=bias, padding='same')(x)

        x = norm_layer(momentum=0.1, epsilon=1e-5)(x)

        if dropout:
            x = tf.keras.layers.Dropout(0.5)(x)

        # Skip is second only because it is more obvious in the architecture image
        x = tf.keras.layers.Concatenate()([x, skip_x])

        return x


class UNet32(UNet):
    def __init__(self, config):
        super().__init__(resolution=config.resolution,
                         input_channels=config.in_channels,
                         output_channels=config.out_channels,
                         filters=config.filters, n_blocks=5,
                         norm_layer=config.norm_layer, dropout=config.dropout)


class UNet64(UNet):
    def __init__(self, config):
        super().__init__(resolution=config.resolution,
                         input_channels=config.in_channels,
                         output_channels=config.out_channels,
                         filters=config.filters, n_blocks=6,
                         norm_layer=config.norm_layer, dropout=config.dropout)


class UNet128(UNet):
    def __init__(self, config):
        super().__init__(resolution=config.resolution,
                         input_channels=config.in_channels,
                         output_channels=config.out_channels,
                         filters=config.filters, n_blocks=7,
                         norm_layer=config.norm_layer, dropout=config.dropout)


class UNet256(UNet):
    def __init__(self, config):
        super().__init__(resolution=config.resolution,
                         input_channels=config.in_channels,
                         output_channels=config.out_channels,
                         filters=config.filters, n_blocks=8,
                         norm_layer=config.norm_layer, dropout=config.dropout)


class UNet512(UNet):
    def __init__(self, config):
        super().__init__(resolution=config.resolution,
                         input_channels=config.in_channels,
                         output_channels=config.out_channels,
                         filters=config.filters, n_blocks=9,
                         norm_layer=config.norm_layer, dropout=config.dropout)


class UNet1024(UNet):
    def __init__(self, config):
        super().__init__(resolution=config.resolution,
                         input_channels=config.in_channels,
                         output_channels=config.out_channels,
                         filters=config.filters, n_blocks=10,
                         norm_layer=config.norm_layer, dropout=config.dropout)
