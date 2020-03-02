from typing import Tuple
import pickle

import numpy as np
from pathlib import Path
import tensorflow as tf

from dataloader.template import DataLoader


class TFRecordDataLoader(DataLoader):
    def __init__(self, dataset: Path, batch_size: int, resolution: int, channels: int):
        super().__init__(dataset, batch_size, resolution, channels)

        self.tfrecord = tf.data.TFRecordDataset([str(self.dataset)])
        self.tfrecord_len = sum([1 for _ in self.tfrecord])

    @property
    def batches(self) -> int:
        return int(self.tfrecord_len / self.batch_size)

    def _get_pair(self, record) -> Tuple[np.ndarray, ...]:
        example = tf.train.Example()
        example.ParseFromString(record.numpy())

        img_A = pickle.loads(example.features.feature["A"].bytes_list.value[0])
        img_B = pickle.loads(example.features.feature["B"].bytes_list.value[0])

        return img_A, img_B

    def get_images(self, n: int) -> Tuple[np.ndarray, ...]:
        shuffled_tfrecord = self.tfrecord.shuffle(buffer_size=int(self.tfrecord_len / 10))

        img_As = np.zeros((n, self.resolution, self.resolution, self.channels))
        img_Bs = np.zeros((n, self.resolution, self.resolution, self.channels))

        for i, record in enumerate(shuffled_tfrecord.take(n)):
            img_A, img_B = self._get_pair(record)

            img_As[i] = img_A
            img_Bs[i] = img_B

        return img_As, img_Bs

    def yield_batch(self) -> Tuple[np.ndarray, ...]:
        img_As = np.zeros((self.batch_size, self.resolution, self.resolution, self.channels))
        img_Bs = np.zeros((self.batch_size, self.resolution, self.resolution, self.channels))

        for records in self.tfrecord.batch(self.batch_size):
            for j, record in enumerate(records):
                img_A, img_B = self._get_pair(record)

                img_As[j] = img_A
                img_Bs[j] = img_B

            yield img_As, img_Bs
