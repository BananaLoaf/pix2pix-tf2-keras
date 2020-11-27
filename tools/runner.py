import itertools

from tqdm import tqdm
import numpy as np

from metaneural.runner import *
from nn.generator import *
from tools.dataloader import Dataloader
import tensorflow.keras.layers.experimental.preprocessing as preprocessing
from nn.discriminator import Discriminator
from tools.config import Config


SAMPLE_N = 5


class GANTrainer(tf.keras.models.Model):
    def __init__(self, G_net, D_net):
        # Fake graph for tensorboard
        real_A = tf.keras.layers.Input(shape=G_net.input_shape[1:], name="real_A")
        real_B = tf.keras.layers.Input(shape=G_net.output_shape[1:], name="real_B")
        fake_B = G_net(real_A)

        real_D = D_net([real_A, real_B])
        fake_D = D_net([real_A, fake_B])

        super().__init__([real_A, real_B], [fake_B, real_D, fake_D])

        self.G_net = G_net
        self.D_net = D_net

    def compile(self, G_optimizer, D_optimizer, g_l1_lambda):
        super(GANTrainer, self).compile(run_eagerly=True)
        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer

        self.g_l1_lambda = g_l1_lambda

        self.REAL_D = tf.ones((1, *self.D_net.output_shape[1:]))
        self.FAKE_D = tf.zeros((1, *self.D_net.output_shape[1:]))

    def augment(self, x: tf.Tensor, seed: int) -> tf.Tensor:
        tf.random.set_seed(seed)  # RandomFlip is stupid, this stays until tf 2.4
        x = preprocessing.RandomFlip("horizontal", seed=seed)(x)
        # img = preprocessing.RandomTranslation(height_factor=(-0.1, 0.1),
        #                                       width_factor=(-0.1, 0.1), seed=seed)(img)
        x = preprocessing.RandomCrop(height=tf.cast(0.9 * x.shape[1], dtype=tf.int32),
                                     width=tf.cast(0.9 * x.shape[2], dtype=tf.int32),
                                     seed=seed)(x)
        # img = preprocessing.RandomZoom((0., -0.1), seed=seed)(img)

        return x

    def resize(self, x):
        x = tf.keras.layers.experimental.preprocessing.Resizing(self.G_net.input_shape[1], self.G_net.input_shape[2])(x)
        x = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)(x)

        return x

    def train_step(self, data):
        real_A, real_B = data
        real_A, real_B = tf.expand_dims(real_A, axis=0), tf.expand_dims(real_B, axis=0)

        seed = int(tf.random.uniform((), maxval=tf.dtypes.int64.max, dtype=tf.dtypes.int64))
        real_A, real_B = self.augment(real_A, seed), self.augment(real_B, seed)
        real_A, real_B = self.resize(real_A), self.resize(real_B)

        with tf.GradientTape() as tape:
            fake_B = self.G_net(real_A, training=True)
            fake_D = self.D_net([real_A, fake_B], training=True)

            G_GAN_loss = tf.reduce_mean(tf.losses.MSE(self.REAL_D, fake_D))
            G_L1 = tf.reduce_mean(tf.losses.MAE(real_B, fake_B)) * self.g_l1_lambda

            grads = tape.gradient(G_GAN_loss + G_L1, self.G_net.trainable_variables)
        self.G_optimizer.apply_gradients(zip(grads, self.G_net.trainable_variables))

        # Train Discriminator
        with tf.GradientTape() as tape:
            real_D_L1 = tf.losses.MSE(self.REAL_D, self.D_net([real_A, real_B], training=True))
            fake_D_L1 = tf.losses.MSE(self.FAKE_D, self.D_net([real_A, fake_B], training=True))
            D_L1 = tf.reduce_mean(real_D_L1 + fake_D_L1)

            grads = tape.gradient(D_L1, self.D_net.trainable_variables)
        self.D_optimizer.apply_gradients(zip(grads, self.D_net.trainable_variables))

        return {
                "D_L1": D_L1,
                "G_L1": G_L1,
                "G_GAN_loss": G_GAN_loss
            }

    def test_step(self, data):
        real_A, real_B = data
        real_A, real_B = tf.expand_dims(real_A, axis=0), tf.expand_dims(real_B, axis=0)
        real_A, real_B = self.resize(real_A), self.resize(real_B)

        fake_B = self.G_net(real_A)
        fake_D = self.D_net([real_A, fake_B])

        G_GAN_loss = tf.reduce_mean(tf.losses.MSE(self.REAL_D, fake_D))
        G_L1 = tf.reduce_mean(tf.losses.MAE(real_B, fake_B)) * self.g_l1_lambda

        real_D_L1 = tf.losses.MSE(self.REAL_D, self.D_net([real_A, real_B]))
        fake_D_L1 = tf.losses.MSE(self.FAKE_D, self.D_net([real_A, fake_B]))
        D_L1 = tf.reduce_mean(real_D_L1 + fake_D_L1)

        ################################################################
        return {
            "D_L1": D_L1,
            "G_L1": G_L1,
            "G_GAN_loss": G_GAN_loss
        }


class CustomRunner(Runner):
    config: Config

    ################################################################
    @Runner.with_strategy
    def init(self) -> Tuple[GANTrainer, Dict[str, tf.keras.optimizers.Optimizer]]:
        G_net: tf.keras.models.Model = GENERATORS[self.config.generator](config=self.config)
        G_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.g_lr,
                                               beta_1=self.config.g_beta1)

        ################################################################
        D_net: tf.keras.models.Model = Discriminator(input_resolution=self.config.resolution,
                                                     a_channels=self.config.in_channels,
                                                     b_channels=self.config.out_channels,
                                                     filters=self.config.filters,
                                                     n_blocks=3,
                                                     norm_layer=self.config.norm_layer)
        D_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.d_lr,
                                               beta_1=self.config.d_beta1)

        ################################################################
        model = GANTrainer(G_net, D_net)
        return model, {"G": G_optimizer, "D": D_optimizer}

    def _loader_generator(self, img_names):
        A = Path(self.config.dataset_a)
        B = Path(self.config.dataset_b)

        for img_name in sorted([p.name for p in A.glob("*.png")]):
            assert A.joinpath(img_name).exists(), f"{A.joinpath(img_name)} does not exist"
            assert B.joinpath(img_name).exists(), f"{B.joinpath(img_name)} does not exist"

        for img_name in img_names:
            img_A = tf.keras.preprocessing.image.load_img(A.joinpath(img_name), grayscale=self.config.in_channels == 1)
            # img_A = tf.expand_dims(tf.keras.preprocessing.image.img_to_array(img_A), axis=0)

            img_B = tf.keras.preprocessing.image.load_img(B.joinpath(img_name), grayscale=self.config.out_channels == 1)
            # img_B = tf.expand_dims(tf.keras.preprocessing.image.img_to_array(img_B), axis=0)

            yield img_A, img_B

    def init_dataloader(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        # Load names
        img_names = sorted([p.name for p in Path(self.config.dataset_a).glob("*.png")])
        train_imgs = img_names[:-round(len(img_names) * self.config.test_split)]
        test_imgs = img_names[-round(len(img_names) * self.config.test_split):]

        # Init datasets
        train_dataset = tf.data.Dataset.from_generator(lambda: self._loader_generator(train_imgs),
                                                       output_types=(tf.float32, tf.float32))
        train_dataset.cache("train_dataset_cache")

        test_dataset = tf.data.Dataset.from_generator(lambda: self._loader_generator(test_imgs),
                                                      output_types=(tf.float32, tf.float32))
        test_dataset.cache("test_dataset_cache")

        return train_dataset, test_dataset

    def quantize_model(self):
        if bool(self.config.q_aware_train[0]):
            import tensorflow_model_optimization as tfmot
            self.model.G_net = tfmot.quantization.keras.quantize_model(self.model.G_net)

        if bool(self.config.q_aware_train[1]):
            import tensorflow_model_optimization as tfmot
            self.model.D_net = tfmot.quantization.keras.quantize_model(self.model.D_net)

    def train(self, resume: bool = False):
        tb = tf.keras.callbacks.TensorBoard(log_dir=self.run_path, histogram_freq=self.config.test_freq,
                                            update_freq="batch", profile_batch=10)
        # TODO checkpoint and config callback
        cb = tf.keras.callbacks.LambdaCallback(on_train_begin=None, on_batch_end=None)

        self.model.compile(G_optimizer=self.optimizer["G"], D_optimizer=self.optimizer["D"],
                           g_l1_lambda=self.config.g_l1_lambda)
        self.model.fit(self.train_dataset, validation_data=self.test_dataset, validation_freq=self.config.test_freq,
                       initial_epoch=self.config.epoch, epochs=self.config.epochs, batch_size=self.config.batch_size,
                       callbacks=[tb, cb])

    ################################################################
    # Saving, snapping, etc
    def summary(self, plot: bool = False):
        self._summary(self.model.G_net, "G", plot)
        self._summary(self.model.D_net, "D", plot)

    # @Runner.with_strategy
    # def train(self, resume: bool = False):
    #     pbar = tqdm(range(self.config.step, self.config.steps + 1), total=self.config.steps)
    #     pbar.update(self.config.step)
    #
    #     for curr_step in pbar:
    #         # Checkpoint
    #         if curr_step % self.config.checkpoint_freq == 0 and not resume:
    #             print("\nSaving checkpoints")
    #             self.snap(curr_step)
    #             self.config.step = curr_step
    #             self.save_config()
    #
    #         # Save sample image
    #         if curr_step % self.config.sample_freq == 0:
    #             self.sample(curr_step)
    #
    #         # Skip last step cuz it's not saved anyway
    #         if curr_step == self.config.steps:
    #             break
    #
    #         ################################################################
    #         # Test
    #         if curr_step % self.config.test_freq == 0:
    #             print("\nTesting")
    #             metrics = self.test()
    #             with self.test_writer.as_default():
    #                 for key, value in metrics.items():
    #                     tf.summary.scalar(key, value, step=curr_step)
    #
    #             # Histograms
    #             for data in self.D_net.trainable_variables:
    #                 tf.summary.histogram(f"D_{data.name}", data, step=curr_step)
    #
    #             for data in self.G_net.trainable_variables:
    #                 tf.summary.histogram(f"G_{data.name}", data, step=curr_step)
    #
    #         # Perform train step
    #         metrics = self.train_step()
    #         for key, value in metrics.items():
    #             tf.summary.scalar(key, value, step=curr_step)
    #
    #         pbar.set_description(f"[Checkpoint in {self.config.checkpoint_freq - (curr_step % self.config.checkpoint_freq)} steps] " +
    #                              " ".join([f"[{key}: {value:.3f}]" for key, value in metrics.items()]))
    #
    #         # Reset state
    #         if resume:
    #             resume = False

    ################################################################
    # Sampling
    # def generate_samples(self, real_As: tf.Tensor, real_Bs: tf.Tensor) -> tf.Tensor:
    #     fake_Bs = self.G_net(real_As)
    #
    #     rows = []
    #     for i in range(real_As.shape[0]):
    #         real_A = tf.keras.layers.experimental.preprocessing.Resizing(height=256, width=256, interpolation="nearest")(real_As[i])
    #         real_B = tf.keras.layers.experimental.preprocessing.Resizing(height=256, width=256, interpolation="nearest")(real_Bs[i])
    #         fake_B = tf.keras.layers.experimental.preprocessing.Resizing(height=256, width=256, interpolation="nearest")(fake_Bs[i])
    #
    #         if real_A.shape[2] == 1:
    #             real_A = tf.concat([real_A, real_A, real_A], axis=2)
    #         if real_B.shape[2] == 1:
    #             real_B = tf.concat([real_B, real_B, real_B], axis=2)
    #         if fake_B.shape[2] == 1:
    #             fake_B = tf.concat([fake_B, fake_B, fake_B], axis=2)
    #
    #         row = tf.concat([real_A, real_B, fake_B], axis=1)
    #         rows.append(row)
    #
    #     return tf.concat(rows, axis=0)
    #
    # def sample(self, step: int):
    #     real_As = []
    #     real_Bs = []
    #
    #     for real_A, real_B in itertools.chain(
    #             self.dataloader.next(batch_size=SAMPLE_N, shuffle=False),
    #             self.dataloader.next(batch_size=SAMPLE_N, shuffle=False, test=True)):
    #         real_As.append(real_A)
    #         real_Bs.append(real_B)
    #
    #     real_As = tf.concat(real_As, axis=0)
    #     real_Bs = tf.concat(real_Bs, axis=0)
    #
    #     assert isinstance(real_As, tf.Tensor)
    #     assert isinstance(real_Bs, tf.Tensor)
    #
    #     ################################################################
    #     rgb_img = self.generate_samples(real_As, real_Bs)
    #     img_path = self.samples_path.joinpath(f"{str(step).zfill(10)}.png")
    #     tf.keras.preprocessing.image.save_img(img_path, rgb_img)
