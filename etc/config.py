from argparse import Namespace
import json
from copy import deepcopy

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


class CF:  # Config Fields
    NAME = "NAME"

    RESOLUTION = "RESOLUTION"
    IN_CHANNELS = "IN_CHANNELS"
    OUT_CHANNELS = "OUT_CHANNELS"
    FILTERS = "FILTERS"
    GENERATOR = "GENERATOR"
    DATALOADER = "DATALOADER"
    DATASET = "DATASET"

    DEVICE = "DEVICE"
    TPU_NAME = "TPU_NAME"
    XLA_JIT = "XLA_JIT"

    LEARNING_RATE = "LEARNING_RATE"
    BETA1 = "BETA1"
    G_L1_LAMBDA = "G_L1_LAMBDA"
    BATCH_SIZE = "BATCH_SIZE"
    STEP = "STEP"
    STEPS = "STEPS"

    SAMPLE_N = "SAMPLE_N"
    SAMPLE_FREQ = "SAMPLE_FREQ"
    CHECKPOINT_FREQ = "CHECKPOINT_FREQ"
    METRICS_FREQ = "METRICS_FREQ"

    SAVE_TF = "SAVE_TF"
    SAVE_TFLITE = "SAVE_TFLITE"
    SAVE_TFLITE_Q = "SAVE_TFLITE_Q"

    @classmethod
    def schema(cls) -> dict:
        self = cls()

        return {
            "type": dict,
            "required": list(filter(
                lambda x: isinstance(x, str), self.__dict__.values()
            )),
            "properties": {
                self.NAME: {"type": str},

                self.RESOLUTION: {"type": int},
                self.IN_CHANNELS: {"type": int},
                self.OUT_CHANNELS: {"type": int},
                self.FILTERS: {"type": int},
                self.GENERATOR: {"type": str},
                self.DATALOADER: {"type": str},
                self.DATASET: {"type": str},

                self.DEVICE: {"type": str},
                self.TPU_NAME: {"type": str},
                self.XLA_JIT: {"type": bool},

                self.LEARNING_RATE: {"type": float},
                self.BETA1: {"type": float},
                self.G_L1_LAMBDA: {"type": float},
                self.BATCH_SIZE: {"type": int},
                self.STEP: {"type": int, "default": 0},
                self.STEPS: {"type": int},

                self.SAMPLE_N: {"type": int},
                self.SAMPLE_FREQ: {"type": int},
                self.CHECKPOINT_FREQ: {"type": int},
                self.METRICS_FREQ: {"type": int},

                self.SAVE_TF: {"type": bool},
                self.SAVE_TFLITE: {"type": bool},
                self.SAVE_TFLITE_Q: {"type": bool},
            }
        }


class Config:
    _editable: tuple = (CF.STEP,)
    _data: dict = {}

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        if key in self._editable:
            self._data[key] = value
        else:
            raise KeyError(f"Item '{key}' is not editable")

    def __delitem__(self, key):
        if key not in CF.__dict__.values():
            del self._data[key]
        else:
            raise NotImplementedError

    def _validate(self):
        schema = CF.schema()
        Validator(schema).validate(self._data)

    def save(self, path: Path):
        with path.open("w") as file:
            json.dump(self._data, file, indent=4)

    @classmethod
    def from_args(cls, args: Namespace):
        self = cls()
        self._data = vars(deepcopy(args))
        for key in list(self._data.keys()):
            try:
                del self[key]
            except NotImplementedError:
                pass

        self._validate()
        return self

    @classmethod
    def from_file(cls, path: Path):
        self = cls()
        with path.open("r") as file:
            self._data = json.load(file)
        self._validate()
        return self
