from tools.config import Config

if __name__ == '__main__':
    config = Config.cli("Pix2Pix Tensorflow 2 Keras implementation")

    from tools.runner import CustomRunner
    CustomRunner.train(config)
