import cv2
from tqdm import tqdm
import numpy as np

from metaneural.runner import *
from nn.generator import *
from tools.dataloader import Dataloader
from nn.discriminator import Discriminator
from tools.config import Config


SAMPLE_N = 5


class CustomRunner(Runner):
    config: Config

    ################################################################
    def init_dataloader(self) -> Dataloader:
        return Dataloader(config=self.config)

    @Runner.with_strategy
    def init_networks(self) -> Tuple[RegistryEntry, ...]:
        self.G_net: tf.keras.models.Model = GENERATORS[self.config.generator](config=self.config)
        self.G_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.g_lr,
                                                    beta_1=self.config.g_beta1)

        ################################################################
        self.D_net: tf.keras.models.Model = Discriminator(input_resolution=self.config.resolution,
                                                          a_channels=self.config.in_channels,
                                                          b_channels=self.config.out_channels,
                                                          filters=self.config.filters,
                                                          n_blocks=3,
                                                          norm_layer=self.config.norm_layer)
        self.D_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.d_lr,
                                                    beta_1=self.config.d_beta1)

        ################################################################
        self.REAL_D = tf.ones((self.config.batch_size, *self.D_net.output_shape[1:]))
        self.FAKE_D = tf.zeros((self.config.batch_size, *self.D_net.output_shape[1:]))

        self.V_REAL_D = tf.ones((self.dataloader.validation_split_size, *self.D_net.output_shape[1:]))
        self.V_FAKE_D = tf.zeros((self.dataloader.validation_split_size, *self.D_net.output_shape[1:]))

        ################################################################
        return (
            RegistryEntry(
                name="G",
                model=self.G_net,
                optimizer=self.G_optimizer
            ),
            RegistryEntry(
                name="D",
                model=self.D_net,
                optimizer=self.D_optimizer
            )
        )

    ################################################################
    def train_step(self) -> dict:
        real_As, real_Bs = self.dataloader.next(batch_size=self.config.batch_size)
        assert isinstance(real_As, tf.Tensor)
        assert isinstance(real_Bs, tf.Tensor)

        ################################################################
        # Train Generator
        with tf.GradientTape() as tape:
            fake_Bs = self.G_net(real_As)
            fake_D = self.D_net([real_As, fake_Bs])

            G_GAN_loss = tf.reduce_mean(tf.losses.MSE(self.REAL_D, fake_D))
            G_L1 = tf.reduce_mean(tf.losses.MAE(real_Bs, fake_Bs)) * self.config.g_l1_lambda

            grads = tape.gradient(G_GAN_loss + G_L1, self.G_net.trainable_variables)
        self.G_optimizer.apply_gradients(zip(grads, self.G_net.trainable_variables))

        # Train Discriminator
        with tf.GradientTape() as tape:
            real_D_L1 = tf.losses.MSE(self.REAL_D, self.D_net([real_As, real_Bs]))
            fake_D_L1 = tf.losses.MSE(self.FAKE_D, self.D_net([real_As, fake_Bs]))
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
    def train(self, resume: bool = False):
        pbar = tqdm(range(self.config.steps))
        pbar.update(self.config.step)

        for curr_step in range(self.config.step, self.config.steps + 1):
            # Checkpoint
            if curr_step % self.config.checkpoint_freq == 0 and not resume:
                print("\nSaving checkpoints")
                self._snap(curr_step)
                self.config.step = curr_step
                self._save_config()

            # Save sample image
            if curr_step % self.config.sample_freq == 0:
                self.sample(curr_step)

            # Skip last step cuz it's not saved anyway
            if curr_step == self.config.steps:
                break

            ################################################################
            # Test
            if curr_step % self.config.test_freq == 0:
                print("\nTesting")
                metrics = self.test()
                with self.test_writer.as_default():
                    for key, value in metrics.items():
                        tf.summary.scalar(key, value, step=curr_step)

                # Histograms
                for data in self.D_net.trainable_variables:
                    tf.summary.histogram(f"D_{data.name}", data, step=curr_step)

                for data in self.G_net.trainable_variables:
                    tf.summary.histogram(f"G_{data.name}", data, step=curr_step)

            # Perform train step
            metrics = self.train_step()
            for key, value in metrics.items():
                tf.summary.scalar(key, value, step=curr_step)

            pbar.set_description(f"[Checkpoint in {self.config.checkpoint_freq - (curr_step % self.config.checkpoint_freq)} steps] " +
                                 " ".join([f"[{key}: {value:.3f}]" for key, value in metrics.items()]))
            pbar.update()

            # Reset state
            if resume:
                resume = False

        pbar.close()

    def test(self) -> dict:
        real_As, real_Bs = self.dataloader.next(batch_size=self.dataloader.validation_split_size, shuffle=False, validate=True)
        assert isinstance(real_As, tf.Tensor)
        assert isinstance(real_Bs, tf.Tensor)

        fake_Bs = self.G_net(real_As)
        fake_D = self.D_net([real_As, fake_Bs])

        G_GAN_loss = tf.reduce_mean(tf.losses.MSE(self.V_REAL_D, fake_D))
        G_L1 = tf.reduce_mean(tf.losses.MAE(real_Bs, fake_Bs)) * self.config.g_l1_lambda

        real_D_L1 = tf.losses.MSE(self.V_REAL_D, self.D_net([real_As, real_Bs]))
        fake_D_L1 = tf.losses.MSE(self.V_FAKE_D, self.D_net([real_As, fake_Bs]))
        D_L1 = tf.reduce_mean(real_D_L1 + fake_D_L1)

        ################################################################
        return {
            "D_L1": D_L1,
            "G_L1": G_L1,
            "G_GAN_loss": G_GAN_loss
        }

    ################################################################
    # Sampling
    def generate_samples(self, real_As: tf.Tensor, real_Bs: tf.Tensor) -> np.ndarray:
        fake_Bs = self.G_net(real_As)

        real_As = tf.cast(tf.math.round((real_As + 1) * 127.5), tf.uint8)
        real_Bs = tf.cast(tf.math.round((real_Bs + 1) * 127.5), tf.uint8)
        fake_Bs = tf.cast(tf.math.round((fake_Bs + 1) * 127.5), tf.uint8)

        rows = []
        for i in range(real_As.shape[0]):
            real_A = real_As[i].numpy()
            real_B = real_Bs[i].numpy()
            fake_B = fake_Bs[i].numpy()

            if real_A.shape[2] == 1:
                real_A = cv2.cvtColor(real_A, cv2.COLOR_GRAY2RGB)
            if real_B.shape[2] == 1:
                real_B = cv2.cvtColor(real_B, cv2.COLOR_GRAY2RGB)
            if fake_B.shape[2] == 1:
                fake_B = cv2.cvtColor(fake_B, cv2.COLOR_GRAY2RGB)

            row = np.hstack([real_A, real_B, fake_B])
            rows.append(row)

        return np.vstack(rows)

    def sample(self, step: int):
        img_As_t, img_Bs_t = self.dataloader.next(batch_size=SAMPLE_N, shuffle=False, no_index=True)
        img_As_v, img_Bs_v = self.dataloader.next(batch_size=SAMPLE_N, shuffle=False, no_index=True, validate=True)
        img_As, img_Bs = tf.concat((img_As_t, img_As_v), 0), tf.concat((img_Bs_t, img_Bs_v), 0)

        assert isinstance(img_As, tf.Tensor)
        assert isinstance(img_Bs, tf.Tensor)

        rgb_img = self.generate_samples(img_As, img_Bs)
        img_path = self.samples_path.joinpath(f"{str(step).zfill(10)}.png")
        cv2.imwrite(str(img_path), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
