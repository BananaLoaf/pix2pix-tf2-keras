from typing import Tuple

from pathlib import Path
import numpy as np
import tensorflow as tf


class DataLoader:
    def __init__(self, dataset: Path, batch_size: int, resolution: int, channels: int):
        self.dataset = dataset
        self.batch_size = batch_size

        self.resolution = resolution
        self.channels = channels

    @property
    def batches(self) -> int:
        raise NotImplementedError
        return 0

    def get_random(self, n: int) -> Tuple[tf.Tensor, ...]:
        raise NotImplementedError

        # Example
        img_As = img_Bs = np.random.rand((n, 512, 512, 3))  # (n, resolution, resolution, channels)
        return tf.convert_to_tensor(img_As), tf.convert_to_tensor(img_Bs)

    def yield_batch(self) -> Tuple[tf.Tensor, ...]:
        raise NotImplementedError

        # Example
        for i in range(self.batches):
            img_As = img_Bs = np.random.rand((self.batch_size, 512, 512, 3))  # (batch_size, resolution, resolution, channels)
            yield tf.convert_to_tensor(img_As), tf.convert_to_tensor(img_Bs)
