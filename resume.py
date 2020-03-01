from argparse import ArgumentParser

from pathlib import Path

from config import Config

if __name__ == '__main__':
    parser = ArgumentParser(description="Pix2Pix tensorflow 2 keras implementation")
    parser.add_argument("path", type=str, help="Path to run directory")
    args = parser.parse_args()

    config = Config.from_file(Path(args.path).joinpath("config.json"))

    from pix2pix import Pix2Pix
    gan = Pix2Pix.resume(config=config, run_directory=Path(args.path))
    try:
        gan.train()
    except KeyboardInterrupt:
        print("Stopping...")
