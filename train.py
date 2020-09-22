from config import Config

if __name__ == '__main__':
    config = Config.cli("Pix2Pix Tensorflow 2 Keras implementation")

    from etc.runner import Runner
    Runner.train(config)
