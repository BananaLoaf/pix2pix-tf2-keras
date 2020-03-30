import os
import datetime
from typing import Optional

from pathlib import Path
import cv2
from tqdm import tqdm

from etc.config import Config
from generator import *
from dataloader import *
from dataloader.template import DataLoader
from etc.discriminator import Discriminator


class Pix2Pix:
    config: Config
    dataloader: DataLoader
    writer: tf.summary.SummaryWriter
    strategy: tf.distribute.Strategy

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

    def __init__(self, config: Config, run_directory: Path):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.config = config
        if self.config.xla_jit:
            os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

        ################################################################
        # Paths
        self.run_path = run_directory
        self.run_path.mkdir(exist_ok=True, parents=True)

        self.samples_path = Path(f"{self.run_path}/samples")
        self.samples_path.mkdir(exist_ok=True)

        self.checkpoints_path = Path(f"{self.run_path}/checkpoints")
        self.checkpoints_path.joinpath("D").mkdir(exist_ok=True, parents=True)
        self.checkpoints_path.joinpath("G").mkdir(exist_ok=True, parents=True)

        self.model_path = Path(f"{self.run_path}/model")
        self.model_path.joinpath("D").mkdir(exist_ok=True, parents=True)
        self.model_path.joinpath("G").mkdir(exist_ok=True, parents=True)

        ################################################################
        # DataLoader, TensorBoard
        DataLoader_class = DATALOADERS[self.config.dataloader]
        self.dataloader = DataLoader_class(config=self.config)
        self.writer = tf.summary.create_file_writer(str(self.run_path))
        self.writer.set_as_default()

        ################################################################
        # Device
        if self.config.use_tpu:
            kwargs = {} if self.config.tpu_name is None else {"tpu": self.config.tpu_name}
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(**kwargs)
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            self.strategy = tf.distribute.experimental.TPUStrategy(resolver)

        else:
            if "GPU" in self.config.device:
                self.strategy = tf.distribute.MirroredStrategy(devices=self.config.device.split(","))
            elif "CPU" in self.config.device:
                self.strategy = tf.distribute.OneDeviceStrategy(device=self.config.device)
            else:
                raise NotImplementedError(f"Wrong device name '{self.config.device}'")

        ################################################################
        self._init_networks()

    def on_device(func):
        """Run function in a strategy context"""
        def wrapper(self, *args, **kwargs):
            return self.strategy.experimental_run_v2(lambda: func(self, *args, **kwargs))

        return wrapper

    def off_device(func):
        """Descope function within other context"""
        def wrapper(*args, **kwargs):
            def descoper(strategy: Optional[tf.distribute.Strategy] = None, *args, **kwargs):
                if strategy is None:
                    return func(*args, **kwargs)
                else:
                    with strategy.scope():
                        return func(*args, **kwargs)

            return tf.distribute.get_replica_context().merge_call(lambda strategy: descoper(strategy, *args, **kwargs))

        return wrapper

    @on_device
    def _init_networks(self):
        Generator_class = GENERATORS[self.config.generator]
        self.G_net = Generator_class(config=self.config)

        self.G_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.g_lr,
                                                    beta_1=self.config.g_beta1)
        self.G_ckpt = tf.train.Checkpoint(optimizer=self.G_optimizer, model=self.G_net)
        self.G_ckpt_manager = tf.train.CheckpointManager(self.G_ckpt, directory=str(self.checkpoints_path.joinpath("G")),
                                                         max_to_keep=self.config.steps, checkpoint_name="G")

        ################################################################
        self.D_net = Discriminator(input_resolution=self.config.resolution,
                                   input_channels=self.config.out_channels,
                                   filters=self.config.filters)

        self.D_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.d_lr,
                                                    beta_1=self.config.d_beta1)
        self.D_ckpt = tf.train.Checkpoint(optimizer=self.D_optimizer, model=self.D_net)
        self.D_ckpt_manager = tf.train.CheckpointManager(self.D_ckpt, directory=str(self.checkpoints_path.joinpath("D")),
                                                         max_to_keep=self.config.steps, checkpoint_name="D")

    @classmethod
    def new_run(cls, config: Config):
        self = cls(config=config,
                   run_directory=Path(f"{config.output}/{config.name}_{datetime.datetime.now().replace(microsecond=0).isoformat()}"))
        self._summary(config.plot)
        self._snap(0)

        self.config.cleanup()
        self._save_config()
        return self

    @classmethod
    def resume(cls, config: Config, run_directory: Path):
        self = cls(config=config,
                   run_directory=run_directory)
        self._restore()
        self._summary()
        return self

    def train(self):
        try:
            self._train()
        except KeyboardInterrupt:
            print("Stopping...")

    @on_device
    def _train(self):
        # Discriminator ground truths
        REAL_D = tf.convert_to_tensor(np.ones((self.dataloader.batch_size, *self.D_net.output_shape[1:])))
        FAKE_D = tf.convert_to_tensor(np.zeros((self.dataloader.batch_size, *self.D_net.output_shape[1:])))

        pbar = tqdm(range(self.config.steps))
        pbar.update(self.config.step)
        for curr_step in range(self.config.step, self.config.steps):
            real_As, Bs = next(self.dataloader)
            assert isinstance(real_As, tf.Tensor)
            assert isinstance(Bs, tf.Tensor)

            ################################################################
            # Save sample image
            if curr_step % self.config.sample_freq == 0:
                self.save_samples(curr_step)

            ################################################################
            # Train step
            # Train Generator
            with tf.GradientTape() as tape:
                fake_As = self.G_net(Bs, training=True)
                fake_D = self.D_net([fake_As, Bs], training=False)

                G_GAN_loss = tf.reduce_mean(tf.losses.MSE(REAL_D, fake_D))
                G_L1 = tf.reduce_mean(tf.losses.MAE(real_As, fake_As)) * self.config.g_l1_lambda

                grads = tape.gradient(G_GAN_loss + G_L1, self.G_net.trainable_variables)
            self.G_optimizer.apply_gradients(zip(grads, self.G_net.trainable_variables))

            # Train Discriminator
            with tf.GradientTape() as tape:
                real_D_L1 = tf.losses.MSE(REAL_D, self.D_net([real_As, Bs], training=True))
                fake_D_L1 = tf.losses.MSE(FAKE_D, self.D_net([fake_As, Bs], training=True))
                D_L1 = tf.reduce_mean(real_D_L1 + fake_D_L1)

                grads = tape.gradient(D_L1, self.D_net.trainable_variables)
            self.D_optimizer.apply_gradients(zip(grads, self.D_net.trainable_variables))

            ################################################################
            metrics = {
                "D_L1": D_L1,
                "G_L1": G_L1,
                "G_GAN_loss": G_GAN_loss
            }

            # TensorBoard
            if curr_step % self.config.metrics_freq == 0:
                for key, value in metrics.items():
                    tf.summary.scalar(key, value, step=curr_step)

            # tqdm
            pbar.set_description(" ".join([f"[{key}: {value:.3f}]" for key, value in metrics.items()]))
            pbar.update()

            # Checkpoint
            if curr_step % self.config.checkpoint_freq == 0 or curr_step == self.config.steps - 1:
                self._snap(curr_step)
                self._save_config(curr_step)
                print("\nCheckpoints saved")

        pbar.close()
        print("Saving models")
        self._save_models()

    # Helpers
    def save_samples(self, step: int):
        with self.dataloader.with_batch_size(self.config.sample_n):
            img_As, img_Bs = next(self.dataloader)
        rgb_img = self.G_net.generate_samples(img_As, img_Bs)
        if rgb_img is not None and type(rgb_img) is np.ndarray:
            img_path = self.samples_path.joinpath(f"{str(step).zfill(10)}.png")
            cv2.imwrite(str(img_path), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

    def _summary(self, plot: bool = False):
        self.D_net.summary()
        self.G_net.summary()

        if plot:
            tf.keras.utils.plot_model(self.D_net, to_file=f"{self.run_path}/D_net.png", show_shapes=True, dpi=64)
            tf.keras.utils.plot_model(self.G_net, to_file=f"{self.run_path}/G_net.png", show_shapes=True, dpi=64)

    @off_device
    def _snap(self, step: int):
        self.D_ckpt_manager.save(step)
        self.G_ckpt_manager.save(step)

    @on_device
    def _restore(self):
        self.D_ckpt.restore(self.D_ckpt_manager.latest_checkpoint)
        self.G_ckpt.restore(self.G_ckpt_manager.latest_checkpoint)

    def _save_models(self):
        # Tensorflow
        if self.config.save_tf:
            self.D_net.save(str(self.model_path.joinpath("D")), save_format="tf")
            self.G_net.save(str(self.model_path.joinpath("G")), save_format="tf")

        # TFLite
        if self.config.save_tflite:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.D_net)
            with self.model_path.joinpath("D.tflite").open("wb") as file:
                file.write(converter.convert())

            converter = tf.lite.TFLiteConverter.from_keras_model(self.G_net)
            with self.model_path.joinpath("G.tflite").open("wb") as file:
                file.write(converter.convert())

        # TFLite quantizised
        if self.config.save_tflite_q:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.D_net)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            with self.model_path.joinpath("Dq.tflite").open("wb") as file:
                file.write(converter.convert())

            converter = tf.lite.TFLiteConverter.from_keras_model(self.G_net)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            with self.model_path.joinpath("Gq.tflite").open("wb") as file:
                file.write(converter.convert())

    def _save_config(self, step: Optional[int] = None):
        if step is not None:
            self.config.step = step + 1
        self.config.write(self.run_path.joinpath("config.json"))
