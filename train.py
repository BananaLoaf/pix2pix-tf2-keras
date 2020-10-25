from tools.config import Config
from tools.runner import CustomRunner

if __name__ == '__main__':
    config = Config.cli("Pix2Pix Tensorflow 2 Keras implementation")
    CustomRunner.new_run(config)
