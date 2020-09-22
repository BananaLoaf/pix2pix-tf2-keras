import tensorflow as tf

from nn.generator import GENERATORS, UNet256
from dataloader import DATALOADERS
from metaneural.config import *


GPU_DEVICES = [dev[-1] for dev in [dev.name for dev in tf.config.list_logical_devices("GPU")]]


class Config(ConfigBuilder):
    name = {ARGS: ["--name"],
            KWARGS: {TYPE: str, REQUIRED: True, HELP: "Model name"}}


    # Device params
    use_tpu = {GROUP_NAME: "Device params",
               ARGS: ["--use-tpu"],
               KWARGS: {ACTION: "store_true",
                        HELP: "Use Google Cloud TPU, if True, --gpu param is ignored (default: %(default)s)"}}
    tpu_name = {GROUP_NAME: "Device params",
                ARGS: ["--tpu-name"],
                KWARGS: {TYPE: str, DEFAULT: None,
                         HELP: "Google Cloud TPU name, if None and flag --use-tpu is set, will try to detect automatically (default: %(default)s)"}}
    devices = {GROUP_NAME: "Device params",
               ARGS: ["--gpu"],
               KWARGS: {TYPE: str, DEFAULT: None,
                        HELP: "Available GPUs: {}, list devices with , as delimiter".format(GPU_DEVICES)}}
    xla_jit = {GROUP_NAME: "Device params",
               ARGS: ["--xla-jit"],
               KWARGS: {ACTION: "store_true",
                        HELP: "XLA Just In Time compilation, https://www.tensorflow.org/xla (default: %(default)s)"}}


    # Model params
    resolution = {GROUP_NAME: "Model params",
                  ARGS: ["-r", "--res"],
                  KWARGS: {TYPE: int, DEFAULT: 256, HELP: "Input and output image resolution (default: %(default)s)"}}
    in_channels = {GROUP_NAME: "Model params",
                   ARGS: ["--in-channels"],
                   KWARGS: {TYPE: int, DEFAULT: 3, HELP: "Generator input image channels (default: %(default)s)"}}
    out_channels = {GROUP_NAME: "Model params",
                    ARGS: ["--out-channels"],
                    KWARGS: {TYPE: int, DEFAULT: 3, HELP: "Generator output image channels, discriminator input image channels (default: %(default)s)"}}
    filters = {GROUP_NAME: "Model params",
               ARGS: ["-f", "--filters"],
               KWARGS: {TYPE: int, DEFAULT: 64, HELP: "Generator filters (default: %(default)s)"}}
    generator = {GROUP_NAME: "Model params",
                 ARGS: ["-g", "--generator"],
                 KWARGS: {TYPE: str, DEFAULT: UNet256.__name__, CHOICES: list(GENERATORS.keys()), HELP: "Generator (default: %(default)s)"}}
    dataloader = {GROUP_NAME: "Model params",
                  ARGS: ["-dl", "--dataloader"],
                  KWARGS: {TYPE: str, REQUIRED: True, CHOICES: list(DATALOADERS.keys()), HELP: "DataLoader"}}
    dataset = {GROUP_NAME: "Model params",
               ARGS: ["-ds", "--dataset"],
               KWARGS: {TYPE: str, REQUIRED: True, HELP: "Path to dataset"}}


    # Training params
    step = {CONSTANT: 0}
    steps = {GROUP_NAME: "Training params",
             ARGS: ["-s", "--steps"],
             KWARGS: {TYPE: int, DEFAULT: 1_000_000, HELP: "Steps (default: %(default)s)"}}
    quantization_training = {GROUP_NAME: "Training params",
                             ARGS: ["-qt", "--quantizised-training"],
                             KWARGS: {ACTION: "store_true",
                                      HELP: "Quantization aware training, https://www.tensorflow.org/model_optimization/guide/quantization/training (default: %(default)s)"}}
    batch_size = {GROUP_NAME: "Training params",
                  ARGS: ["-b", "--batch-size"],
                  KWARGS: {TYPE: int, DEFAULT: 2, HELP: "Batch size (default: %(default)s)"}}
    checkpoint_freq = {GROUP_NAME: "Training params",
                       ARGS: ["-cf", "--checkpoint-freq"],
                       KWARGS: {TYPE: int, DEFAULT: 10_000,
                                HELP: "Checkpoint frequency in steps (default: %(default)s)"}}
    # Custom
    g_l1_lambda = {GROUP_NAME: "Training params",
                   ARGS: ["--G_L1-lambda"],
                   KWARGS: {TYPE: float, DEFAULT: 50.0, HELP: "G_L1 lambda (default: %(default)s)"}}
    sample_freq = {GROUP_NAME: "Training params",
                   ARGS: ["-sf", "--sample-freq"],
                   KWARGS: {TYPE: int, DEFAULT: 100, HELP: "Sampling frequency in steps (default: %(default)s)"}}
    sample_n = {GROUP_NAME: "Training params",
                ARGS: ["-sn", "--sample-n"],
                KWARGS: {TYPE: int, DEFAULT: 6, HELP: "Amount of samples (default: %(default)s)"}}


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


    # Saving params
    save_tflite = {GROUP_NAME: "Saving params",
                   ARGS: ["--tflite"],
                   KWARGS: {ACTION: "store_true", DEFAULT: False, HELP: "Save as tflite model"}}
    save_tflite_q = {GROUP_NAME: "Saving params",
                     ARGS: ["--tflite-q"],
                     KWARGS: {ACTION: "store_true", DEFAULT: False, HELP: "Save as quantizised tflite model"}}
