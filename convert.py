from pathlib import Path

from tools.config import Config, ConverterConfig
from tools.runner import CustomRunner

if __name__ == '__main__':
    converter_config = ConverterConfig.cli("Pix2Pix Tensorflow 2 Keras implementation")
    config = Config.load(Path(converter_config.path).joinpath("config.json"))
    CustomRunner.convert(config=config, converter_config=converter_config)
