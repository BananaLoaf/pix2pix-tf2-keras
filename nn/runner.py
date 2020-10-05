import os
import datetime
from typing import Optional, Tuple, Type

from pathlib import Path
import cv2
from tqdm import tqdm

from metaneural.runner import *
from nn.generator import *
from nn.dataloader import Dataloader
from nn.discriminator import Discriminator
from nn.config import Config


class CustomRunner(Runner):
    config: Config

    ################################################################
    def _init_dataloader(self) -> Dataloader:
        return Dataloader(batch_size=self.config.batch_size, config=self.config)

    @Runner.with_strategy
    def _init_networks(self) -> dict:
        self.G_net: tf.keras.models.Model = GENERATORS[self.config.generator](config=self.config)
        self.G_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.g_lr,
                                                    beta_1=self.config.g_beta1)

        G_checkpoint_path = self.checkpoints_path.joinpath("G")
        G_checkpoint_path.mkdir(exist_ok=True, parents=True)
        self.G_ckpt = tf.train.Checkpoint(optimizer=self.G_optimizer, model=self.G_net)
        self.G_ckpt_manager = tf.train.CheckpointManager(self.G_ckpt, directory=G_checkpoint_path,
                                                         max_to_keep=self.config.steps, checkpoint_name=G_checkpoint_path.name)

        ################################################################
        self.D_net: tf.keras.models.Model = Discriminator(input_resolution=self.config.resolution,
                                                          input_channels=self.config.out_channels,
                                                          filters=self.config.filters)
        self.D_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.d_lr,
                                                    beta_1=self.config.d_beta1)

        D_checkpoint_path = self.checkpoints_path.joinpath("D")
        D_checkpoint_path.mkdir(exist_ok=True, parents=True)
        self.D_ckpt = tf.train.Checkpoint(optimizer=self.D_optimizer, model=self.D_net)
        self.D_ckpt_manager = tf.train.CheckpointManager(self.D_ckpt, directory=D_checkpoint_path,
                                                         max_to_keep=self.config.steps, checkpoint_name=D_checkpoint_path.name)

        self.REAL_D = tf.ones((self.dataloader.batch_size, *self.D_net.output_shape[1:]))
        self.FAKE_D = tf.zeros((self.dataloader.batch_size, *self.D_net.output_shape[1:]))

        ################################################################
        return {
            "G": {
                MODEL: self.G_net,
                OPTIMIZER: self.G_optimizer,
                CHECKPOINT: self.G_ckpt,
                CHECKPOINT_MANAGER: self.G_ckpt_manager
            },
            "D": {
                MODEL: self.D_net,
                OPTIMIZER: self.D_optimizer,
                CHECKPOINT: self.D_ckpt,
                CHECKPOINT_MANAGER: self.D_ckpt_manager
            }
        }

    ################################################################
    def train_step(self) -> dict:
        real_As, Bs = next(self.dataloader)
        assert isinstance(real_As, tf.Tensor)
        assert isinstance(Bs, tf.Tensor)

        ################################################################
        # Train Generator
        with tf.GradientTape() as tape:
            fake_As = self.G_net(Bs, training=True)
            fake_D = self.D_net([fake_As, Bs], training=False)

            G_GAN_loss = tf.reduce_mean(tf.losses.MSE(self.REAL_D, fake_D))
            G_L1 = tf.reduce_mean(tf.losses.MAE(real_As, fake_As)) * self.config.g_l1_lambda

            grads = tape.gradient(G_GAN_loss + G_L1, self.G_net.trainable_variables)
        self.G_optimizer.apply_gradients(zip(grads, self.G_net.trainable_variables))

        # Train Discriminator
        with tf.GradientTape() as tape:
            real_D_L1 = tf.losses.MSE(self.REAL_D, self.D_net([real_As, Bs], training=True))
            fake_D_L1 = tf.losses.MSE(self.FAKE_D, self.D_net([fake_As, Bs], training=True))
            D_L1 = tf.reduce_mean(real_D_L1 + fake_D_L1)

            grads = tape.gradient(D_L1, self.D_net.trainable_variables)
        self.D_optimizer.apply_gradients(zip(grads, self.D_net.trainable_variables))

        ################################################################
        return {
            "D_L1": D_L1,
            "G_L1": G_L1,
            "G_GAN_loss": G_GAN_loss
        }

    @Runner.with_strategy
    def _train(self):
        pbar = tqdm(range(self.config.steps))
        pbar.update(self.config.step)

        for curr_step in range(self.config.step, self.config.steps):
            self.config.step = curr_step

            # Checkpoint
            if curr_step % self.config.checkpoint_freq == 0:
                self._snap(curr_step)
                self._save_config()
                print("\nCheckpoints saved")

            # Save sample image
            if curr_step % self.config.sample_freq == 0:
                self.save_samples(curr_step)

            # Perform train step
            metrics = self.train_step()

            for key, value in metrics.items():
                tf.summary.scalar(key, value, step=curr_step)

            pbar.set_description(" ".join([f"[{key}: {value:.3f}]" for key, value in metrics.items()]))
            pbar.update()

        pbar.close()
        print("Saving models")
        self._save_models()

    ################################################################
    # Sampling
    def generate_samples(self, real_As: tf.Tensor, real_Bs: tf.Tensor) -> np.ndarray:
        n = real_As.shape[0]
        fake_As = self.G_net(real_Bs).numpy()

        rgb_img = np.hstack([real_Bs[0], real_As[0], fake_As[0]])
        for row in range(1, n):
            rgb_img = np.vstack([
                rgb_img,
                np.hstack([real_Bs[row], real_As[row], fake_As[row]])
            ])
        rgb_img = ((rgb_img + 1) * 127.5).astype(np.uint8)

        return rgb_img

    def save_samples(self, step: int):
        with self.dataloader.with_batch_size(self.config.sample_n):
            img_As, img_Bs = next(self.dataloader)

        rgb_img = self.generate_samples(img_As, img_Bs)
        if rgb_img is not None and type(rgb_img) is np.ndarray:
            self.samples_path.mkdir(exist_ok=True, parents=True)
            img_path = self.samples_path.joinpath(f"{str(step).zfill(10)}.png")
            cv2.imwrite(str(img_path), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))