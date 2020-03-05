import tensorflow as tf
import numpy as np


class Generator(tf.keras.models.Model):
    def __init__(self, resolution: int, input_channels: int, output_channels: int, filters: int, n_blocks: int):
        self.filters = filters
        self.n_blocks = n_blocks

        input_layer = tf.keras.layers.Input(shape=(resolution, resolution, input_channels))
        output_layer = do_something(input_layer, channels=output_channels)

        super().__init__(input_layer, output_layer)

    def generate_samples(self, real_As: tf.Tensor, real_Bs: tf.Tensor) -> np.ndarray:
        """Must return RGB image if possible, else None"""
        n = real_As.shape[0]
        fake_As = self(real_Bs).numpy()

        rgb_img = np.hstack([real_Bs[0], real_As[0], fake_As[0]])
        for row in range(1, n):
            rgb_img = np.vstack([
                rgb_img,
                np.hstack([real_Bs[row], real_As[row], fake_As[row]])
            ])
        rgb_img = ((rgb_img + 1) * 127.5).astype(np.uint8)

        return rgb_img
