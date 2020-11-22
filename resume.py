from pathlib import Path

from tools.config import Config, ResumeConfig
from tools.runner import CustomRunner

if __name__ == '__main__':
    resume_config = ResumeConfig.cli("Pix2Pix Tensorflow 2 Keras implementation")
    config = Config.load(Path(resume_config.path).joinpath("config.json"))
    CustomRunner.resume(config=config, resume_config=resume_config)
