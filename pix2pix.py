import os
import datetime
from typing import Optional

import cv2
from tqdm import tqdm

from etc.config import *
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

        if self.config[XLA]:
            os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

        self.run_path = run_directory
        self.run_path.mkdir(exist_ok=True, parents=True)

        self._init_strategy()
        self._init_fields()
        self._init_networks()

    def strategy_scope(func):
        """Run function in a strategy context"""
        def wrapper(self, *args, **kwargs):
            self.strategy.experimental_run_v2(lambda: func(self, *args, **kwargs))

        return wrapper

    def descope(func):
        """Descope function within other context"""
        def wrapper(*args, **kwargs):
            def descoper(strategy: Optional[tf.distribute.Strategy] = None, *args, **kwargs):
                if strategy is None:
                    func(*args, **kwargs)
                else:
                    with strategy.scope():
                        func(*args, **kwargs)

            tf.distribute.get_replica_context().merge_call(lambda strategy: descoper(strategy, *args, **kwargs))

        return wrapper

    def _init_strategy(self):
        if "cpu" in self.config[DEVICE]:
            self.strategy = tf.distribute.OneDeviceStrategy(device=self.config[DEVICE])
        elif "gpu" in self.config[DEVICE]:
            self.strategy = tf.distribute.MirroredStrategy(devices=self.config[DEVICE].split(";"))
        else:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            self.strategy = tf.distribute.experimental.TPUStrategy(resolver)

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

    @strategy_scope
    def _init_networks(self):
        """Create nets from config"""
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
                                     output_channels=self.config[OUT_CHANNELS],
                                     filters=self.config[FILTERS])

        self.G_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config[LEARNING_RATE], beta_1=self.config[BETA1])
        self.G_ckpt = tf.train.Checkpoint(optimizer=self.G_optimizer, model=self.G_net)
        self.G_ckpt_manager = tf.train.CheckpointManager(self.G_ckpt, directory=str(self.checkpoints_path.joinpath("G")),
                                                         max_to_keep=self.config[EPOCHS], checkpoint_name="G")

    @classmethod
    def new_run(cls, config: Config):
        self = cls(config=config,
                   run_directory=Path(f"result/{config[NAME]}_{datetime.datetime.now().replace(microsecond=0).isoformat()}"))
        self._summary()
        self._snap(0)
        self._save_config()
        return self

    @classmethod
    def resume(cls, config: Config, run_directory: Path):
        self = cls(config=config,
                   run_directory=run_directory)
        self._restore()
        self._summary(plot=False)
        return self

    @strategy_scope
    def train(self):
        # Discriminator ground truths
        REAL_D = tf.convert_to_tensor(np.ones((self.dataloader.batch_size, *self.D_net.output_shape[1:])))
        FAKE_D = tf.convert_to_tensor(np.zeros((self.dataloader.batch_size, *self.D_net.output_shape[1:])))

        for epoch in range(self.config[EPOCH], self.config[EPOCHS]):
            ################################################################
            # Train on the dataset
            pbar = tqdm(range(self.dataloader.batches))
            for batch_i, (real_As, Bs) in enumerate(self.dataloader.yield_batch()):
                ################################################################
                # Save sample images
                if self.config[STEP] % self.config[SAMPLE_FREQ] == 0:
                    img_As, img_Bs = self.dataloader.get_records(self.config[SAMPLE_N])
                    rgb_img = self.G_net.generate_samples(img_As, img_Bs)
                    if rgb_img is not None:
                        img_path = self.samples_path.joinpath(f"{str(self.config[STEP]).zfill(10)}.png")
                        cv2.imwrite(str(img_path), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

                ################################################################
                #  Train Discriminator
                fake_As = self.G_net(Bs)
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
                    G_L1 = tf.reduce_mean(tf.losses.MAE(real_As, fake_As)) * self.config[G_L1_LAMBDA]

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
            # Epoch ended
            self.config[EPOCH] = epoch + 1
            pbar.close()

            # Checkpoint
            if epoch % self.config[CHECKPOINT_FREQ] == 0:
                print("Saving checkpoint")
                self._snap(epoch)
                self._save_config()

        print("Saving models")
        self._save_models()

    # Helpers
    def _summary(self, plot: bool = True):
        self.D_net.summary()
        self.G_net.summary()

        if plot:
            tf.keras.utils.plot_model(self.D_net, to_file=f"{self.run_path}/D_net.png", show_shapes=True, dpi=64)
            tf.keras.utils.plot_model(self.G_net, to_file=f"{self.run_path}/G_net.png", show_shapes=True, dpi=64)

    @descope
    def _snap(self, epoch: int):
        self.D_ckpt_manager.save(epoch)
        self.G_ckpt_manager.save(epoch)

    def _save_models(self):
        self.D_net.save(str(self.model_path.joinpath("D")), save_format="tf")
        self.G_net.save(str(self.model_path.joinpath("G")), save_format="tf")

    def _save_config(self):
        self.config.save(self.run_path.joinpath("config.json"))

    def _restore(self):
        self.D_ckpt.restore(self.D_ckpt_manager.latest_checkpoint)
        self.G_ckpt.restore(self.G_ckpt_manager.latest_checkpoint)
