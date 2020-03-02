from argparse import Namespace
import json

from jsonschema import Draft7Validator, validators, ValidationError
from pathlib import Path


def set_defaults(validator, properties, instance, schema):
    for property, subschema in properties.items():
        if "default" in subschema:
            instance.setdefault(property, subschema["default"])

    for error in Draft7Validator.VALIDATORS["properties"](validator, properties, instance, schema):
        yield error


def check_types(validator, types, instance, schema):
    if not isinstance(instance, types):
        yield ValidationError(f"{instance} is not of type {types}")


Validator = validators.extend(Draft7Validator, {"properties": set_defaults, "type": check_types})


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
G_L1_LAMBDA = "G_L1_LAMBDA"
BATCH_SIZE = "BATCH_SIZE"
STEP = "STEP"
EPOCH = "EPOCH"
EPOCHS = "EPOCHS"

SAMPLE_N = "SAMPLE_N"
SAMPLE_FREQ = "SAMPLE_FREQ"
CHECKPOINT_FREQ = "CHECKPOINT_FREQ"


CONFIG_SCHEMA = {
    "type": dict,
    "required": [
        NAME,

        RESOLUTION,
        IN_CHANNELS,
        OUT_CHANNELS,
        FILTERS,
        GENERATOR,
        DATALOADER,
        DATASET,

        LEARNING_RATE,
        BETA1,
        G_L1_LAMBDA,
        BATCH_SIZE,
        EPOCHS,

        SAMPLE_N,
        SAMPLE_FREQ,
        CHECKPOINT_FREQ,
    ],
    "properties": {
        NAME: {"type": str},

        RESOLUTION: {"type": int},
        IN_CHANNELS: {"type": int},
        OUT_CHANNELS: {"type": int},
        FILTERS: {"type": int},
        GENERATOR: {"type": str},
        DATALOADER: {"type": str},
        DATASET: {"type": str},

        LEARNING_RATE: {"type": float},
        BETA1: {"type": float},
        G_L1_LAMBDA: {"type": float},
        BATCH_SIZE: {"type": int},
        STEP: {"type": int, "default": 0},
        EPOCH: {"type": int, "default": 0},
        EPOCHS: {"type": int},

        SAMPLE_N: {"type": int},
        SAMPLE_FREQ: {"type": int},
        CHECKPOINT_FREQ: {"type": int}
    }
}


class Config:
    EDITABLE: tuple = (STEP, EPOCH)
    _data: dict = {}

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        if key in self.EDITABLE:
            self._data[key] = value
        else:
            raise KeyError(f"Item '{key}' is not editable")

    def _validate(self):
        Validator(CONFIG_SCHEMA).validate(self._data)

    def save(self, path: Path):
        with path.open("w") as file:
            json.dump(self._data, file, indent=4)

    @classmethod
    def from_args(cls, args: Namespace):
        self = cls()
        self._data = vars(args)
        self._validate()
        return self

    @classmethod
    def from_file(cls, path: Path):
        self = cls()
        with path.open("r") as file:
            self._data = json.load(file)
        self._validate()
        return self
