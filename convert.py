from metaneural import convert

from tools.config import CustomConfig
from tools.runner import CustomRunner

if __name__ == '__main__':
    convert(CustomConfig, CustomRunner, "Pix2Pix Tensorflow 2 Keras implementation")
