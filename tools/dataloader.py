from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers.experimental.preprocessing as preprocessing

from tools.config import Config


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

    @property
    def train_split_size(self) -> int:
        return len(self.train_imgs)

    @property
    def test_split_size(self) -> int:
        return len(self.test_imgs)

    def load_pipeline(self, img_path: Path, grayscale: bool, seed: int, augment: bool):
        img = tf.keras.preprocessing.image.load_img(img_path, grayscale=grayscale)
        img = tf.expand_dims(tf.keras.preprocessing.image.img_to_array(img), axis=0)

        if augment:
            tf.random.set_seed(seed)  # RandomFlip is stupid, this stays until tf 2.4
            img = preprocessing.RandomFlip("horizontal", seed=seed)(img)
            # img = preprocessing.RandomTranslation(height_factor=(-0.1, 0.1),
            #                                       width_factor=(-0.1, 0.1), seed=seed)(img)
            img = preprocessing.RandomCrop(height=tf.cast(0.9 * img.shape[1], dtype=tf.int32),
                                           width=tf.cast(0.9 * img.shape[2], dtype=tf.int32),
                                           seed=seed)(img)
            # img = preprocessing.RandomZoom((0., -0.1), seed=seed)(img)

        img = tf.keras.layers.experimental.preprocessing.Resizing(self.resolution, self.resolution)(img)
        img = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)(img)

        return img

    def next(self, batch_size: int, shuffle: bool = True, test: bool = False, augment: bool = False):
        # Which slice
        if test:
            src_imgs = self.test_imgs
        else:
            src_imgs = self.train_imgs

        # Shuffle
        if shuffle:
            src_imgs = np.random.permutation(src_imgs)

        ################################################################
        for img_name in src_imgs[:batch_size]:
            if augment:
                seed = int(tf.random.uniform((), maxval=tf.dtypes.int64.max, dtype=tf.dtypes.int64))
            else:
                seed = 0

            img_A = self.load_pipeline(self.A.joinpath(img_name), grayscale=self.in_channels == 1, seed=seed, augment=augment)
            img_B = self.load_pipeline(self.B.joinpath(img_name), grayscale=self.out_channels == 1, seed=seed, augment=augment)

            yield img_A, img_B
