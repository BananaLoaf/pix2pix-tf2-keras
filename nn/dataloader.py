from pathlib import Path

from metaneural.dataloader import DefaultDataloader
from nn.config import Config


class Dataloader(DefaultDataloader):
    def __init__(self, batch_size: int, config: Config):
        super().__init__(batch_size)
        self.resolution = config.resolution
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels

        self.A = Path(config.dataset_a)
        self.B = Path(config.dataset_b)

        self.image_names = [p.name for p in self.A.glob("*.png")]

    def __len__(self):
        return len(self.image_names)

    def __next__(self):
        pass
