from argparse import ArgumentParser

from pathlib import Path

from tools.config import Config

if __name__ == '__main__':
    parser = ArgumentParser(description="Pix2Pix tensorflow 2 keras implementation")
    parser.add_argument("path", type=str, help="Path to run directory")
    args = parser.parse_args()

    config = Config.load(Path(args.path).joinpath("config.json"))

    from tools.runner import CustomRunner
    CustomRunner.resume(config=config, run_directory=Path(args.path))
