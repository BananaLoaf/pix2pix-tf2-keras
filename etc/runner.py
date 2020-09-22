import os
import datetime
from typing import Optional, Tuple

from pathlib import Path
import cv2
from tqdm import tqdm

from config import Config
from nn.generator import *
from dataloader import *
from dataloader.template import DataLoader
from nn.discriminator import Discriminator


MODEL = "MODEL"
OPTIMIZER = "OPTIMIZER"
CHECKPOINT = "CHECKPOINT"
CHECKPOINT_MANAGER = "CHECKPOINT_MANAGER"


class Runner:
    def __init__(self, config: Config, run_directory: Path):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.config = config
        if config.xla_jit:
            os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

        ################################################################
        # Paths
        self.run_path = run_directory
        self.samples_path = self.run_path.joinpath("samples")
        self.checkpoints_path = self.run_path.joinpath("checkpoints")
        self.model_path = self.run_path.joinpath("model")

        ################################################################
        # DataLoader
        self.dataloader: DataLoader = DATALOADERS[config.dataloader](config=config)

        ################################################################
        self._strategy = self._init_strategy()
        self._model_registry = self._init_networks()

    @property
    def model_registry(self) -> Tuple[str, tf.keras.models.Model, tf.keras.optimizers.Optimizer, tf.train.Checkpoint, tf.train.CheckpointManager]:
        for model_name, reg in self._model_registry.items():
            yield model_name, reg[MODEL], reg[OPTIMIZER], reg[CHECKPOINT], reg[CHECKPOINT_MANAGER]

    ################################################################
    # https://www.tensorflow.org/api_docs/python/tf/distribute
    def _init_strategy(self) -> tf.distribute.Strategy:
        if self.config.use_tpu:
            kwargs = {} if self.config.tpu_name is None else {"tpu": self.config.tpu_name}
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(**kwargs)
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            return tf.distribute.experimental.TPUStrategy(resolver)

        else:
            all_devices = [dev.name for dev in tf.config.list_logical_devices()]

            if self.config.devices is not None:
                devices = [f"/device:{'XLA_' if self.config.xla_jit else ''}GPU:{dev}" for dev in self.config.devices.split(",")]
                for device in devices:
                    assert device in all_devices, f"Invalid device {device}"

                return tf.distribute.MirroredStrategy(devices=devices)
            else:
                device = f"/device:{'XLA_' if self.config.xla_jit else ''}CPU:0"
                assert device in all_devices, f"Invalid device {device}"

                return tf.distribute.OneDeviceStrategy(device=device)

    def with_strategy(func):
        """Run function in a strategy context"""
        def wrapper(self, *args, **kwargs):
            return self._strategy.experimental_run_v2(lambda: func(self, *args, **kwargs))

        return wrapper

    def merge(func):
        """
        Merge args across replicas and run merge_fn in a cross-replica context. Whatever that means.
        https://www.tensorflow.org/api_docs/python/tf/distribute/ReplicaContext
        """
        def wrapper(*args, **kwargs):
            def descoper(strategy: Optional[tf.distribute.Strategy] = None, *args2, **kwargs2):
                if strategy is None:
                    return func(*args2, **kwargs2)
                else:
                    with strategy.scope():
                        return func(*args2, **kwargs2)

            return tf.distribute.get_replica_context().merge_call(descoper, args, kwargs)

        return wrapper

    ################################################################
    @with_strategy
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
    @classmethod
    def train(cls, config: Config):
        self = cls(config=config,
                   run_directory=Path(f"runs/{config.name}_{datetime.datetime.now().replace(microsecond=0).isoformat()}"))
        self._summary(plot=True)
        self._snap(0)
        self._save_config()

        try:
            self._train()
        except KeyboardInterrupt:
            print("Saving and stopping...")
            self._save_models()

    @classmethod
    def resume(cls, config: Config, run_directory: Path):
        self = cls(config=config,
                   run_directory=run_directory)
        self._restore()
        self._summary()
        return self

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

    @with_strategy
    def _train(self):
        pbar = tqdm(range(self.config.steps))
        pbar.update(self.config.step)

        for curr_step in range(self.config.step, self.config.steps):
            # Save sample image
            if curr_step % self.config.sample_freq == 0:
                self.save_samples(curr_step)

            # Perform train step
            metrics = self.train_step()

            for key, value in metrics.items():
                tf.summary.scalar(key, value, step=curr_step)

            pbar.set_description(" ".join([f"[{key}: {value:.3f}]" for key, value in metrics.items()]))
            pbar.update()

            # Checkpoint
            if curr_step % self.config.checkpoint_freq == 0:
                self._snap(curr_step)
                self._save_config(curr_step)
                print("\nCheckpoints saved")

        pbar.close()
        print("Saving models")
        self._snap(self.config.steps - 1)
        self._save_config(self.config.steps - 1)
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

    ################################################################
    # Saving, snapping, etc
    def _summary(self, plot: bool = False):
        for model_name, model, optimizer, checkpoint, checkpoint_manager in self.model_registry:
            model.summary()
            if plot:
                img_path = self.run_path.joinpath(model_name)
                tf.keras.utils.plot_model(model, to_file=str(img_path), show_shapes=True, dpi=64)

    @merge
    def _snap(self, step: int):
        for model_name, model, optimizer, checkpoint, checkpoint_manager in self.model_registry:
            checkpoint_manager.save(step)

    @with_strategy
    def _restore(self):
        for model_name, model, optimizer, checkpoint, checkpoint_manager in self.model_registry:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)

    def _save_models(self):
        for model_name, model, optimizer, checkpoint, checkpoint_manager in self.model_registry:
            # Tensorflow
            model.save(str(self.model_path.joinpath(model_name)), save_format="tf")

            # TFLite
            if self.config.save_tflite:
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                with self.model_path.joinpath(f"{model_name}.tflite").open("wb") as file:
                    file.write(converter.convert())

            # TFLite quantizised
            if self.config.save_tflite_q:
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                with self.model_path.joinpath(f"{model_name}_q.tflite").open("wb") as file:
                    file.write(converter.convert())

    def _save_config(self, step: Optional[int] = None):
        if step is not None:  # Increase step so that it starts from it when resuming the training
            self.config.step = step + 1
        self.config.save(self.run_path.joinpath("config.json"))
