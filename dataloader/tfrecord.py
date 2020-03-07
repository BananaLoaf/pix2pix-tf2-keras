from typing import Tuple, Optional
import pickle

import numpy as np
from pathlib import Path
import tensorflow as tf

from dataloader.template import DataLoader


class TFRecordDataLoader(DataLoader):
    def __init__(self, dataset: Path, batch_size: int, resolution: int, channels: int):
        super().__init__(dataset, batch_size, resolution, channels)

        self.tfrecord = tf.data.TFRecordDataset([str(self.dataset)])
        self._len = sum([1 for _ in self.tfrecord])

    def __len__(self):
        return self._len

    @property
    def batches(self) -> int:
        return int(len(self) / self.batch_size)

    def _record2img(self, record) -> Tuple[np.ndarray, ...]:
        example = tf.train.Example()
        example.ParseFromString(record.numpy())

        # i = example.features.feature["i"].int64_list.value[0]  # for debug purposes
        img_A = pickle.loads(example.features.feature["A"].bytes_list.value[0])
        img_B = pickle.loads(example.features.feature["B"].bytes_list.value[0])

        return img_A, img_B

    def get_random(self, n: Optional[int] = None) -> Tuple[tf.Tensor, ...]:
        if n is None:
            n = self.batch_size
        assert n <= len(self), "n is bigger than dataset"

        img_As = np.zeros((n, self.resolution, self.resolution, self.channels))
        img_Bs = np.zeros((n, self.resolution, self.resolution, self.channels))

        for records in self.tfrecord.shuffle(buffer_size=len(self)+1).batch(n, drop_remainder=True):
            for i, record in enumerate(records):
                img_A, img_B = self._record2img(record)

                img_As[i] = img_A
                img_Bs[i] = img_B

            return tf.convert_to_tensor(img_As), tf.convert_to_tensor(img_Bs)
