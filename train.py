from metaneural import train

from tools.config import CustomConfig
from tools.runner import CustomRunner

if __name__ == '__main__':
    train(CustomConfig, CustomRunner, "Pix2Pix Tensorflow 2 Keras implementation")