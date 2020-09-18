from argparse import ArgumentParser
import json
from typing import Tuple, Any

import tensorflow as tf
from pathlib import Path

from generator import GENERATORS, UNet256
from dataloader import DATALOADERS


GPU_DEVICES = [dev[-1] for dev in [dev.name for dev in tf.config.list_logical_devices("GPU")]]


ARGS = "ARGS"
KWARGS = "KWARGS"
GROUP_NAME = "GROUP_NAME"
# EXCLUSIVE_GROUP = "EXCLUSIVE_GROUP"
CONSTANT = "CONSTANT"
SAVE = "SAVE"

TYPE = "type"
ACTION = "action"
REQUIRED = "required"
DEFAULT = "default"
CHOICES = "choices"
HELP = "help"


class ConfigBuilder:
    def __init__(self):
        self._field_scheme = {}

        # Move all schemes in a protected dict
        for field, scheme in vars(self.__class__).items():
            if not (field.startswith("__") and field.endswith("__")) and isinstance(scheme, dict):
                setattr(self, field, None)
                self._field_scheme[field] = self.set_defaults(scheme)

    @staticmethod
    def set_defaults(scheme: dict) -> dict:
        """Set all default values for a scheme"""
        scheme.setdefault(SAVE, True)
        return scheme

    def get_field_scheme_value(self) -> Tuple[str, dict, Any]:
        """Fetches all available fields, schemes and values"""
        for field, scheme in self._field_scheme.items():
            try:
                value = getattr(self, field)
                self._field_scheme[field] = scheme
                yield field, scheme, value

            except AttributeError:
                pass

    @classmethod
    def from_cli(cls, description: str = ""):
        self = cls()

        ################################################################
        # Create parser
        parser = ArgumentParser(description=description)
        groups = {}

        for field, scheme, value in self.get_field_scheme_value():
            # Set constants and skip
            if CONSTANT in scheme.keys():
                setattr(self, field, scheme[CONSTANT])
                continue

            # Create group and set as target for new argument
            if GROUP_NAME in scheme.keys():
                groups.setdefault(scheme[GROUP_NAME], parser.add_argument_group(scheme[GROUP_NAME]))
                target_parser = groups[scheme[GROUP_NAME]]

            else:
                target_parser = parser

            target_parser.add_argument(*scheme[ARGS], **scheme[KWARGS], dest=field)

        ################################################################
        # Parse
        args = parser.parse_args()
        for field, value in vars(args).items():
            setattr(self, field, value)

        return self

    @classmethod
    def from_file(cls, path: Path):
        self = cls()

        with path.open("r") as file:
            data = json.load(file)

        for field, scheme, value in self.get_field_scheme_value():
            if scheme[SAVE]:
                try:
                    setattr(self, field, data[field])
                except KeyError:
                    raise KeyError(f"Config is missing required key '{field}'")

        self.cleanup()
        return self

    def to_json(self) -> dict:
        data = {}
        for field, scheme, value in self.get_field_scheme_value():
            if scheme[SAVE]:
                data[field] = getattr(self, field)
        return data

    def write(self, path: Path):
        with path.open("w") as file:
            json.dump(self.to_json(), file, indent=4)

    def cleanup(self):
        for field, scheme, value in self.get_field_scheme_value():
            if not scheme[SAVE]:
                delattr(self, field)
                delattr(self.__class__, field)


class Config(ConfigBuilder):
    """
    This config implementation allows to easily trace params usage with the help of IDE

    Examples:
    name = {GROUP_NAME: "Model",                                        # Not required
            ARGS: ["--name"],                                           # Required
            KWARGS: {TYPE: str, REQUIRED: True, HELP: "Model name"}},   # Required
            SAVE: False                                                 # Saving in config.json, (default: True)

    # Does not provide cli param, just exists
    step = {CONSTANT: 0,
            SAVE; False}  #

    # Not used
    device = {GROUP_NAME: "Device params",
              EXCLUSIVE_GROUP: [
                  {ARGS: ["--cpu"],
                   KWARGS: {TYPE: str, DEFAULT: "/device:CPU:0", CHOICES: [dev.name for dev in tf.config.list_logical_devices("CPU")], HELP: "CPU (default: %(default)s)"}},
                  {ARGS: ["--gpu"],
                   KWARGS: {TYPE: str, HELP: "GPUs"}}
              ],
              REQUIRED: False}  # Only used with EXCLUSIVE_GROUP, if not required, one of elements in a group must have DEFAULT value (default: True)
    """

    name = {ARGS: ["--name"],
            KWARGS: {TYPE: str, REQUIRED: True, HELP: "Model name"}}
    plot = {ARGS: ["--plot"],
            KWARGS: {ACTION: "store_true", HELP: "Plot network architectures into png files, requires pydot and graphviz installed (default: %(default)s)"},
            SAVE: False}

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

    # Device params
    use_tpu = {GROUP_NAME: "Device params",
               ARGS: ["--use-tpu"],
               KWARGS: {ACTION: "store_true", HELP: "Use Google Cloud TPU, if True, --gpu param is ignored (default: %(default)s)"}}
    tpu_name = {GROUP_NAME: "Device params",
                ARGS: ["--tpu-name"],
                KWARGS: {TYPE: str, DEFAULT: None, HELP: "Google Cloud TPU name, if None and flag --use-tpu is set, will try to detect automatically (default: %(default)s)"}}
    devices = {GROUP_NAME: "Device params",
               ARGS: ["--gpu"],
               KWARGS: {TYPE: str, DEFAULT: None, HELP: f"Available GPUs: {GPU_DEVICES}, list devices with , as delimiter"}}
    xla_jit = {GROUP_NAME: "Device params",
               ARGS: ["--xla-jit"],
               KWARGS: {ACTION: "store_true", HELP: "XLA Just In Time compilation, does not fully support UpSampling2D layer in TF 2.1.0 and may never will, https://www.tensorflow.org/xla (default: %(default)s)"}}

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

    # Training params
    g_l1_lambda = {GROUP_NAME: "Training params",
                   ARGS: ["--G_L1-lambda"],
                   KWARGS: {TYPE: float, DEFAULT: 50.0, HELP: "G_L1 lambda (default: %(default)s)"}}
    batch_size = {GROUP_NAME: "Training params",
                  ARGS: ["-b", "--batch-size"],
                  KWARGS: {TYPE: int, DEFAULT: 2, HELP: "Batch size (default: %(default)s)"}}
    step = {CONSTANT: 0}
    steps = {GROUP_NAME: "Training params",
             ARGS: ["-s", "--steps"],
             KWARGS: {TYPE: int, DEFAULT: 1_000_000, HELP: "Steps (default: %(default)s)"}}
    sample_freq = {GROUP_NAME: "Training params",
                   ARGS: ["-sf", "--sample-freq"],
                   KWARGS: {TYPE: int, DEFAULT: 100, HELP: "Sampling frequency in steps (default: %(default)s)"}}
    sample_n = {GROUP_NAME: "Training params",
                ARGS: ["-sn", "--sample-n"],
                KWARGS: {TYPE: int, DEFAULT: 6, HELP: "Amount of samples (default: %(default)s)"}}
    checkpoint_freq = {GROUP_NAME: "Training params",
                       ARGS: ["-cf", "--checkpoint-freq"],
                       KWARGS: {TYPE: int, DEFAULT: 10_000, HELP: "Checkpoint frequency in steps (default: %(default)s)"}}

    # Saving params
    save_tflite = {GROUP_NAME: "Saving params",
                   ARGS: ["--tflite"],
                   KWARGS: {ACTION: "store_true", DEFAULT: False, HELP: "Save as tflite model"}}
    save_tflite_q = {GROUP_NAME: "Saving params",
                     ARGS: ["--tflite-q"],
                     KWARGS: {ACTION: "store_true", DEFAULT: False, HELP: "Save as quantizised tflite model"}}


if __name__ == '__main__':
    c = Config.from_cli()
    # c.cleanup()
    # c.write(Path("config.json"))

    c2 = Config.from_file(Path("config.json"))
