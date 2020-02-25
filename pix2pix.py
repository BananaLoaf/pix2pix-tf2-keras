import os
import datetime
from typing import Tuple, List
import json

import cv2
from pathlib import Path
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from generator import *
from dataloader import *
from dataloader.template import DataLoader
from discriminator import Discriminator


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


NAME = "NAME"

RESOLUTION = "RESOLUTION"
IN_CHANNELS = "IN_CHANNELS"
OUT_CHANNELS = "OUT_CHANNELS"
FILTERS = "FILTERS"
GENERATOR = "GENERATOR"
DATALOADER = "DATALOADER"
DATASET = "DATASET"

LEARNING_RATE = "LEARNING_RATE"
BETA1 = "BETA1"
LAMBDA = "LAMBDA"
BATCH_SIZE = "BATCH_SIZE"
STEP = "STEP"
EPOCH = "EPOCH"
EPOCHS = "EPOCHS"


class Pix2Pix:
    config: dict
    dataloader: DataLoader
    writer: tf.summary.SummaryWriter

    run_path: Path
    samples_path: Path
    model_path: Path

    D_net: tf.keras.models.Model
    D_optimizer: tf.keras.optimizers.Adam
    D_ckpt: tf.train.Checkpoint
    D_ckpt_manager: tf.train.CheckpointManager

    G_net: tf.keras.models.Model
    G_optimizer: tf.keras.optimizers.Adam
    G_ckpt: tf.train.Checkpoint
    G_ckpt_manager: tf.train.CheckpointManager

    def __init__(self, config: dict, run_directory: Path):
        self.config = config

        self.run_path = run_directory
        self.run_path.mkdir()

        self._init_fields()

    def _summary(self, plot: bool = True):
        self.D_net.summary()
        self.G_net.summary()

        if plot:
            tf.keras.utils.plot_model(self.D_net, to_file=f"{self.run_path}/D_net.png", show_shapes=True, dpi=64)
            tf.keras.utils.plot_model(self.G_net, to_file=f"{self.run_path}/G_net.png", show_shapes=True, dpi=64)

    def _restore(self):
        self.D_ckpt.restore(self.D_ckpt_manager.latest_checkpoint)
        self.G_ckpt.restore(self.G_ckpt_manager.latest_checkpoint)

    def _snap(self, epoch: int):
        self.D_ckpt_manager.save(epoch)
        self.G_ckpt_manager.save(epoch)

    def _save(self):
        self.D_net.save(str(self.model_path.joinpath("D")), save_format="tf")
        self.G_net.save(str(self.model_path.joinpath("G")), save_format="tf")

    @classmethod
    def new_run(cls, config: dict):
        self = cls(config=config,
                   run_directory=Path(f"result/{config[NAME]}_{datetime.datetime.now().replace(microsecond=0).isoformat()}"))
        self._summary()
        return self

    @classmethod
    def resume(cls, run_directory: Path):
        assert run_directory.exists(), "Run directory doesn't exist!"
        assert run_directory.is_dir(), "Not a directory!"
        assert run_directory.joinpath("config.json").exists(), "Config file not found!"
        assert len(list(run_directory.joinpath("checkpoints").glob("*"))) > 0, "No checkpoints found!"

        with run_directory.joinpath("config.json").open("r") as file:
            config = json.load(file)

        self = cls(config=config,
                   run_directory=run_directory)
        self._restore()
        return self

    def _init_fields(self):
        """Create fields from config"""
        DataLoader_class = DATALOADERS[self.config[DATALOADER]]
        self.dataloader = DataLoader_class(dataset=Path(self.config[DATASET]),
                                           batch_size=self.config[BATCH_SIZE],
                                           resolution=self.config[RESOLUTION],
                                           channels=self.config[IN_CHANNELS])
        self.writer = tf.summary.create_file_writer(str(self.run_path))

        ################################################################
        self.samples_path = Path(f"{self.run_path}/samples")
        self.samples_path.mkdir(exist_ok=True)

        self.checkpoints_path = Path(f"{self.run_path}/checkpoints")
        self.checkpoints_path.joinpath("D").mkdir(exist_ok=True, parents=True)
        self.checkpoints_path.joinpath("G").mkdir(exist_ok=True, parents=True)

        self.model_path = Path(f"{self.run_path}/model")
        self.model_path.joinpath("D").mkdir(exist_ok=True, parents=True)
        self.model_path.joinpath("G").mkdir(exist_ok=True, parents=True)

        ################################################################
        self.D_net = Discriminator(input_resolution=self.config[RESOLUTION],
                                   input_channels=self.config[IN_CHANNELS],
                                   filters=self.config[FILTERS])

        self.D_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config[LEARNING_RATE], beta_1=self.config[BETA1])
        self.D_ckpt = tf.train.Checkpoint(optimizer=self.D_optimizer, model=self.D_net)
        self.D_ckpt_manager = tf.train.CheckpointManager(self.D_ckpt, directory=str(self.checkpoints_path.joinpath("D")),
                                                         max_to_keep=self.config[EPOCHS], checkpoint_name="D")

        ################################################################
        Generator_class = GENERATORS[self.config[GENERATOR]]
        self.G_net = Generator_class(resolution=self.config[RESOLUTION],
                                     input_channels=self.config[IN_CHANNELS],
                                     output_channels=self.config[OUT_CHANNELS])

        self.G_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config[LEARNING_RATE], beta_1=self.config[BETA1])
        self.G_ckpt = tf.train.Checkpoint(optimizer=self.G_optimizer, model=self.G_net)
        self.G_ckpt_manager = tf.train.CheckpointManager(self.G_ckpt, directory=str(self.checkpoints_path.joinpath("G")),
                                                         max_to_keep=self.config[EPOCHS], checkpoint_name="G")

    def train(self):
        # Discriminator ground truths
        REAL_D = np.ones((self.dataloader.batch_size, *self.D_net.output_shape[1:]))
        FAKE_D = np.zeros((self.dataloader.batch_size, *self.D_net.output_shape[1:]))

        for epoch in range(self.config[EPOCH], self.config[EPOCHS]):
            ################################################################
            # Train on the dataset
            pbar = tqdm(range(self.dataloader.batches))
            for batch_i, (real_As, Bs) in enumerate(self.dataloader.yield_batch()):
                ################################################################
                #  Train Discriminator
                fake_As = self.G_net.predict(Bs)
                with tf.GradientTape() as tape:
                    real_D_L1 = tf.losses.MSE(REAL_D, self.D_net([real_As, Bs], training=True))
                    fake_D_L1 = tf.losses.MSE(FAKE_D, self.D_net([fake_As, Bs], training=True))
                    D_L1 = tf.reduce_mean(real_D_L1 + fake_D_L1)

                    grads = tape.gradient(D_L1, self.D_net.trainable_variables)
                self.D_optimizer.apply_gradients(zip(grads, self.D_net.trainable_variables))

                ################################################################
                #  Train Generator
                with tf.GradientTape() as tape:
                    fake_As = self.G_net(Bs, training=True)
                    with tape.stop_recording():
                        fake_D = self.D_net([fake_As, Bs], training=False)

                    G_GAN_loss = tf.reduce_mean(tf.losses.MSE(REAL_D, fake_D))
                    G_L1 = tf.reduce_mean(tf.losses.MAE(real_As, fake_As)) * self.config[LAMBDA]

                    grads = tape.gradient(G_GAN_loss + G_L1, self.G_net.trainable_variables)
                self.G_optimizer.apply_gradients(zip(grads, self.G_net.trainable_variables))

                ################################################################
                # Write stuff
                with self.writer.as_default():
                    tf.summary.scalar("D_L1", D_L1, step=self.config[STEP])
                    tf.summary.scalar("G_L1", G_L1, step=self.config[STEP])
                    tf.summary.scalar("G_GAN_loss", G_GAN_loss, step=self.config[STEP])
                self.writer.flush()

                ################################################################
                # Status info
                pbar.set_description(f"[Epoch {epoch + 1}/{self.config[EPOCHS]}] "
                                     f"[Step: {self.config[STEP]}] "
                                     f"[D_L1: {D_L1:.3f}] "
                                     f"[G_GAN_loss: {G_GAN_loss:.3f}, G_L1: {G_L1:.3f}] ")
                pbar.update()
                self.config[STEP] += 1

            ################################################################
            # Save sample images
            if epoch % 1 == 0:
                self.generate_sample(6)

            # Checkpoint
            if epoch % 10 == 0:
                print("Saving checkpoint")
                self._snap(epoch)

            ################################################################
            # Epoch ended
            pbar.close()

            self.config[EPOCH] = epoch + 1
            with self.run_path.joinpath("config.json").open("w") as file:
                json.dump(self.config, file, indent=4)

        self._save()

    def generate_sample(self, n: int):
        real_As, real_Bs = self.dataloader.get_images(n)
        fake_As = self.G_net.predict(real_Bs)

        rgb_img = np.hstack([real_Bs[0], real_As[0], fake_As[0]])
        for row in range(1, n):
            rgb_img = np.vstack([
                rgb_img,
                np.hstack([real_Bs[row], real_As[row], fake_As[row]])
            ])
        rgb_img = ((rgb_img + 1) * 127.5).astype(np.uint8)

        img_path = self.samples_path.joinpath(f"{str(self.config[STEP]).zfill(10)}.png")
        cv2.imwrite(str(img_path), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
