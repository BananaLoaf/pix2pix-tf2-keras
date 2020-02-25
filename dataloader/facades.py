from dataloader.template import DataLoader

from typing import Tuple

from pathlib import Path
import numpy as np
import cv2


class Facades(DataLoader):
    def __init__(self, dataset: Path, batch_size: int, resolution: int, channels: int):
        super().__init__(dataset, batch_size, resolution, channels)

        self.paths = [path.with_suffix("") for path in self.dataset.glob("*.png")]

    @property
    def batches(self) -> int:
        return int(len(self.paths) / self.batch_size)

    def imread(self, path: Path) -> np.ndarray:
        mode = cv2.IMREAD_GRAYSCALE if self.channels == 1 else cv2.IMREAD_COLOR
        img = cv2.imread(str(path), mode)
        img = cv2.resize(img, (self.resolution, self.resolution))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float) / 127.5 - 1

    def _get_pair(self, img_path) -> Tuple[np.ndarray, ...]:
        return self.imread(img_path.with_suffix(".jpg")), self.imread(img_path.with_suffix(".png"))

    def get_images(self, n: int) -> Tuple[np.ndarray, ...]:
        img_As = np.zeros((n, self.resolution, self.resolution, self.channels))
        img_Bs = np.zeros((n, self.resolution, self.resolution, self.channels))

        for i, img_path in enumerate(np.random.choice(self.paths, size=n)):
            img_A, img_B = self._get_pair(img_path)

            img_As[i] = img_A
            img_Bs[i] = img_B

        return img_As, img_Bs

    def yield_batch(self) -> Tuple[np.ndarray, ...]:
        for i in range(self.batches):
            img_As = np.zeros((self.batch_size, self.resolution, self.resolution, self.channels))
            img_Bs = np.zeros((self.batch_size, self.resolution, self.resolution, self.channels))

            for i, img_path in enumerate(self.paths[i * self.batch_size:(i + 1) * self.batch_size]):
                img_A, img_B = self._get_pair(img_path)

                img_As[i] = img_A
                img_Bs[i] = img_B

            yield img_As, img_Bs
