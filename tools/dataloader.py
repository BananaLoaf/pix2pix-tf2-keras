from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf

from tools.config import Config


channels2code = {
    1: cv2.COLOR_BGR2GRAY,
    3: cv2.COLOR_BGR2RGB,
}


class Dataloader:
    def __init__(self, config: Config):
        self.resolution = config.resolution
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels

        self.A = Path(config.dataset_a)
        self.B = Path(config.dataset_b)

        self.img_names = sorted([p.name for p in self.A.glob("*.png")])
        for img_name in self.img_names:
            assert self.A.joinpath(img_name).exists(), f"{self.A.joinpath(img_name)} does not exist"
            assert self.B.joinpath(img_name).exists(), f"{self.B.joinpath(img_name)} does not exist"

        self.train_imgs = self.img_names[:-int(len(self.img_names) * config.test_split)]
        self.test_imgs = self.img_names[-int(len(self.img_names) * config.test_split):]

        self.ti = 0
        self.vi = 0

    @property
    def train_split_size(self) -> int:
        return len(self.train_imgs)

    @property
    def test_split_size(self) -> int:
        return len(self.test_imgs)

    def inc(self, value: int, max_val: int):
        if value == max_val:
            value = 0
        else:
            value += 1
        return value

    def next(self, batch_size: int, shuffle: bool = True, test: bool = False, no_index: bool = False):
        # Which slice
        if test:
            src_imgs = self.test_imgs
        else:
            src_imgs = self.train_imgs

        # What index to use
        if not no_index:
            if test:
                i = self.vi
            else:
                i = self.ti
        else:
            i = 0

        # Shuffle
        if shuffle:
            src_imgs = np.random.permutation(src_imgs)

        ################################################################
        img_As = np.zeros((batch_size, self.resolution, self.resolution, self.in_channels))
        img_Bs = np.zeros((batch_size, self.resolution, self.resolution, self.out_channels))

        for bi in range(batch_size):
            img_name = src_imgs[i]

            img_A = cv2.imread(str(self.A.joinpath(img_name)), cv2.IMREAD_COLOR)
            img_B = cv2.imread(str(self.B.joinpath(img_name)), cv2.IMREAD_COLOR)

            img_A = cv2.cvtColor(img_A, channels2code[self.in_channels])
            img_B = cv2.cvtColor(img_B, channels2code[self.out_channels])

            img_A = cv2.resize(img_A, (self.resolution, self.resolution))
            img_B = cv2.resize(img_B, (self.resolution, self.resolution))

            if len(img_A.shape) == 2:
                img_A = np.reshape(img_A, img_A.shape + (1,))
            if len(img_B.shape) == 2:
                img_B = np.reshape(img_B, img_B.shape + (1,))

            img_As[bi] = img_A
            img_Bs[bi] = img_B

            i = self.inc(i, len(src_imgs) - 1)

        ################################################################
        # Update index
        if not no_index:
            if test:
                self.vi = i
            else:
                self.ti = i

        return tf.convert_to_tensor(img_As) / 127.5 - 1, tf.convert_to_tensor(img_Bs) / 127.5 - 1
