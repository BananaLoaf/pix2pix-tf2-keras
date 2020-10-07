from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf

from metaneural.dataloader import DefaultDataloader
from tools.config import Config


channels2code = {
    1: cv2.COLOR_BGRA2GRAY,
    3: cv2.COLOR_BGRA2RGB,
    4: cv2.COLOR_BGRA2RGBA
}


class Dataloader(DefaultDataloader):
    def __init__(self, batch_size: int, config: Config):
        super().__init__(batch_size)
        self.resolution = config.resolution
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels

        self.A = Path(config.dataset_a)
        self.B = Path(config.dataset_b)

        self.img_names = sorted([p.name for p in self.A.glob("*.png")])
        # for img_name in self.img_names:
        #     assert self.A.joinpath(img_name).exists(), f"{self.A.joinpath(img_name)} does not exist"
        #     assert self.B.joinpath(img_name).exists(), f"{self.B.joinpath(img_name)} does not exist"

        self.train_imgs = self.img_names[:-int(len(self.img_names) * config.validation_split)]
        self.valid_imgs = self.img_names[-int(len(self.img_names) * config.validation_split):]

    def __next__(self):
        img_As = np.zeros((self.batch_size, self.resolution, self.resolution, self.in_channels))
        img_Bs = np.zeros((self.batch_size, self.resolution, self.resolution, self.out_channels))

        for i, train_img_name in enumerate(np.random.permutation(self.train_imgs)[:self.batch_size]):
            img_A = cv2.imread(str(self.A.joinpath(train_img_name)), cv2.IMREAD_UNCHANGED)
            img_B = cv2.imread(str(self.B.joinpath(train_img_name)), cv2.IMREAD_UNCHANGED)

            img_A = cv2.cvtColor(img_A, channels2code[self.in_channels])
            img_B = cv2.cvtColor(img_B, channels2code[self.out_channels])

            img_A = cv2.resize(img_A, (self.resolution, self.resolution))
            img_B = cv2.resize(img_B, (self.resolution, self.resolution))

            if len(img_A.shape) == 2:
                img_A = np.reshape(img_A, img_A.shape + (1,))
            if len(img_B.shape) == 2:
                img_B = np.reshape(img_B, img_B.shape + (1, ))

            img_As[i] = img_A
            img_Bs[i] = img_B

        return tf.convert_to_tensor(img_As), tf.convert_to_tensor(img_Bs)