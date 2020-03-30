from typing import Tuple, Optional

from pathlib import Path
import numpy as np
import tensorflow as tf


class DataLoader:
    def __init__(self, config):
        self.dataset = config.dataset
        self.batch_size = config.batch_size

        self.resolution = config.resolution
        self.channels = config.in_channels

    def __len__(self):
        raise NotImplementedError

    def with_batch_size(dl, n: int):
        class Context:
            old_batch_size = dl.batch_size
            new_batch_size = n

            def __enter__(self):
                dl.batch_size = self.new_batch_size

            def __exit__(self, exc_type, exc_val, exc_tb):
                dl.batch_size = self.old_batch_size

        return Context()

    def __next__(self) -> Tuple[tf.Tensor, ...]:
        """
        Example:
            >>> assert self.batch_size <= len(self), "n is bigger than dataset"
            >>> img_As = img_Bs = np.random.rand((self.batch_size, self.resolution, self.resolution, self.channels))  # (batch_size, resolution, resolution, channels)
            >>> return tf.convert_to_tensor(img_As), tf.convert_to_tensor(img_Bs)
        """
        raise NotImplementedError
