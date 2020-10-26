from typing import List, Optional

import tensorflow as tf


class MinMaxNormalizationLayer(tf.keras.layers.Layer):
    def __init__(self, min: float, max: float, newmin: float, newmax: float, round: bool = False, *args, **kwargs):
        super(MinMaxNormalizationLayer, self).__init__(*args, **kwargs)
        self.min = min
        self.max = max
        self.newmin = newmin
        self.newmax = newmax

        self.round = round

    def call(self, input, **kwargs):
        res = (input - self.min)/(self.max - self.min) * (self.newmax - self.newmin) + self.newmin

        if self.round:
            res = tf.math.round(res)

        return res


class SwitchLayer(tf.keras.layers.Layer):
    def __init__(self, run: Optional[tf.keras.layers.Layer] = None, train: Optional[tf.keras.layers.Layer] = None, *args, **kwargs):
        super(SwitchLayer, self).__init__(*args, **kwargs)

        self.run = run
        self.train = train

    def call(self, input, **kwargs):
        kwargs.setdefault("training", False)

        if kwargs["training"]:
            if self.train is None:
                return input
            else:
                return self.train(input)

        else:
            if self.run is None:
                return input
            else:
                return self.run(input)


class UNet(tf.keras.models.Model):
    def __init__(self, resolution: int, input_channels: int, output_channels: int, filters: int, n_blocks: int):
        self.filters = filters
        self.n_blocks = n_blocks

        input_layer = tf.keras.layers.Input(shape=(resolution, resolution, input_channels))

        # If not training, take proper image and normalize it
        # If training, image should be already normalized
        processed_input_layer = SwitchLayer(
            run=MinMaxNormalizationLayer(min=0., max=255., newmin=-1., newmax=1.)
        )(input_layer)

        layers = self.encoder(input=processed_input_layer)
        decoder_layer = self.decoder(layers, channels=output_channels)

        # If not training, make a proper image
        # If training, leave image normalized
        output_layer = SwitchLayer(
            run=MinMaxNormalizationLayer(min=-1., max=1., newmin=0., newmax=255., round=True)
        )(decoder_layer)

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


class UNet32(UNet):
    def __init__(self, config):
        super().__init__(resolution=config.resolution,
                         input_channels=config.in_channels,
                         output_channels=config.out_channels,
                         filters=config.filters, n_blocks=5)


class UNet64(UNet):
    def __init__(self, config):
        super().__init__(resolution=config.resolution,
                         input_channels=config.in_channels,
                         output_channels=config.out_channels,
                         filters=config.filters, n_blocks=6)


class UNet128(UNet):
    def __init__(self, config):
        super().__init__(resolution=config.resolution,
                         input_channels=config.in_channels,
                         output_channels=config.out_channels,
                         filters=config.filters, n_blocks=7)


class UNet256(UNet):
    def __init__(self, config):
        super().__init__(resolution=config.resolution,
                         input_channels=config.in_channels,
                         output_channels=config.out_channels,
                         filters=config.filters, n_blocks=8)


class UNet512(UNet):
    def __init__(self, config):
        super().__init__(resolution=config.resolution,
                         input_channels=config.in_channels,
                         output_channels=config.out_channels,
                         filters=config.filters, n_blocks=9)


class UNet1024(UNet):
    def __init__(self, config):
        super().__init__(resolution=config.resolution,
                         input_channels=config.in_channels,
                         output_channels=config.out_channels,
                         filters=config.filters, n_blocks=10)
