import tensorflow as tf

from nn.generator import GENERATORS, UNet256
from metaneural.config import *


GPU_DEVICES = [dev[-1] for dev in [dev.name for dev in tf.config.list_logical_devices("GPU")]]


class Config(DefaultConfig):
    # Model params
    resolution = {GROUP_NAME: "Model params",
                  ARGS: ["-r", "--res"],
                  KWARGS: {TYPE: int, DEFAULT: 256, HELP: "Input and output image resolution (default: %(default)s)"}}
    in_channels = {GROUP_NAME: "Model params",
                   ARGS: ["--in-channels"],
                   KWARGS: {TYPE: int, DEFAULT: 3, CHOICES: [1, 3], HELP: "Generator input image channels (default: %(default)s)"}}
    out_channels = {GROUP_NAME: "Model params",
                    ARGS: ["--out-channels"],
                    KWARGS: {TYPE: int, DEFAULT: 3, CHOICES: [1, 3], HELP: "Generator output image channels, discriminator input image channels (default: %(default)s)"}}
    filters = {GROUP_NAME: "Model params",
               ARGS: ["-f", "--filters"],
               KWARGS: {TYPE: int, DEFAULT: 64, HELP: "Generator filters (default: %(default)s)"}}
    generator = {GROUP_NAME: "Model params",
                 ARGS: ["-g", "--generator"],
                 KWARGS: {TYPE: str, DEFAULT: UNet256.__name__, CHOICES: list(GENERATORS.keys()), HELP: "Generator (default: %(default)s)"}}
    norm_layer = {GROUP_NAME: "Model params",
                  ARGS: ["-nl", "--norm-layer"],
                  KWARGS: {TYPE: str, DEFAULT: "BatchNormalization", CHOICES: ["BatchNormalization", "InstanceNormalization"], HELP: "Normalization layer, bias used if InstanceNormalization (default: %(default)s)"}}
    dropout = {GROUP_NAME: "Model params",
               ARGS: ["--dropout"],
               KWARGS: {ACTION: "store_true", HELP: "Use Dropout layer (default: %(default)s)"}}

    # Dataloader params
    batch_size = {GROUP_NAME: "Dataloader params",
                  ARGS: ["-b", "--batch-size"],
                  KWARGS: {TYPE: int, DEFAULT: 1, HELP: "Batch size (default: %(default)s)"}}
    dataset_a = {GROUP_NAME: "Dataloader params",
                 ARGS: ["-da", "--dataset-a"],
                 KWARGS: {TYPE: str, REQUIRED: True, HELP: "Path to dataset A, only PNG"}}
    dataset_b = {GROUP_NAME: "Dataloader params",
                 ARGS: ["-db", "--dataset-b"],
                 KWARGS: {TYPE: str, REQUIRED: True, HELP: "Path to dataset B, only PNG"}}


    # Custom training params
    g_l1_lambda = {GROUP_NAME: "Training params",
                   ARGS: ["--G-L1-lambda"],
                   KWARGS: {TYPE: float, DEFAULT: 50.0, HELP: "G_L1 lambda (default: %(default)s)"}}


    # Optimizer params
    g_lr = {GROUP_NAME: "Optimizer params",
            ARGS: ["--g-lr"],
            KWARGS: {TYPE: float, DEFAULT: 0.0002, HELP: "Generator optimizer learning rate (default: %(default)s)"}}
    g_beta1 = {GROUP_NAME: "Optimizer params",
               ARGS: ["--g-beta1"],
               KWARGS: {TYPE: float, DEFAULT: 0.5, HELP: "Generator optimizer beta1 (default: %(default)s)"}}

    d_lr = {GROUP_NAME: "Optimizer params",
            ARGS: ["--d-lr"],
            KWARGS: {TYPE: float, DEFAULT: 0.0002, HELP: "Discriminator optimizer learning rate (default: %(default)s)"}}
    d_beta1 = {GROUP_NAME: "Optimizer params",
               ARGS: ["--d-beta1"],
               KWARGS: {TYPE: float, DEFAULT: 0.5, HELP: "Discriminator optimizer beta1 (default: %(default)s)"}}


Config.devices[KWARGS][HELP].format(GPU_DEVICES)
Config.q_aware_train[KWARGS][DEFAULT] = [0, 0]
