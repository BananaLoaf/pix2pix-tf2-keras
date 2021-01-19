from metaneural import resume
from tools.config import CustomConfig
from tools.runner import CustomRunner

if __name__ == '__main__':
    resume(CustomConfig, CustomRunner, "Pix2Pix Tensorflow 2 Keras implementation")
