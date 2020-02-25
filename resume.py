from argparse import ArgumentParser

from pix2pix import *

if __name__ == '__main__':
    parser = ArgumentParser(description="Pix2Pix tensorflow 2 keras implementation")
    parser.add_argument("path", type=str, help="Path to run directory")
    args = parser.parse_args()

    gan = Pix2Pix.resume(run_directory=Path(args.path))
    gan.train()
