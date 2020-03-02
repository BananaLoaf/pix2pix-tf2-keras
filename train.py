from argparse import ArgumentParser

from config import *
from generator import *
from dataloader import *

if __name__ == '__main__':
    parser = ArgumentParser(description="Pix2Pix tensorflow 2 keras implementation")
    parser.add_argument("--name", type=str, required=True, help="Model name",
                        dest=NAME)
    # Model params
    parser.add_argument("-r", "--res", type=int, default=256, help="Input and output image resolution (default: %(default)s)",
                        dest=RESOLUTION)
    parser.add_argument("--in-channels", type=int, default=3, help="Input image channels (default: %(default)s)",
                        dest=IN_CHANNELS)
    parser.add_argument("--out-channels", type=int, default=3, help="Output image channels (default: %(default)s)",
                        dest=OUT_CHANNELS)
    parser.add_argument("-f", "--filters", type=int, default=64, help="Generator filters (default: %(default)s)",
                        dest=FILTERS)
    parser.add_argument("-g", "--generator", type=str, default=UNet256.__name__, choices=list(GENERATORS.keys()), help="Generator (default: %(default)s)",
                        dest=GENERATOR)
    parser.add_argument("-d", "--dataloader", type=str, required=True, choices=list(DATALOADERS.keys()), help="DataLoader",
                        dest=DATALOADER)
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset",
                        dest=DATASET)
    # Training params
    parser.add_argument("--lr", type=float, default=0.0002, help="Adam Learning rate (default: %(default)s)",
                        dest=LEARNING_RATE)
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam Beta1 (default: %(default)s)",
                        dest=BETA1)
    parser.add_argument("--G_L1-lambda", type=float, default=50.0, help="G_L1 lambda (default: %(default)s)",
                        dest=G_L1_LAMBDA)
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="Batch size (default: %(default)s)",
                        dest=BATCH_SIZE)
    parser.add_argument("-e", "--epochs", type=int, default=1_000, help="Epochs (default: %(default)s)",
                        dest=EPOCHS)
    # Training options
    parser.add_argument("-sn", "--sample-n", type=int, default=6, help="Amount of samples (default: %(default)s)",
                        dest=SAMPLE_N)
    parser.add_argument("-sf", "--sample-freq", type=int, default=100, help="Sampling frequency in steps (default: %(default)s)",
                        dest=SAMPLE_FREQ)
    parser.add_argument("-cf", "--checkpoint-freq", type=int, default=10, help="Checkpoint frequency in epochs (default: %(default)s)",
                        dest=CHECKPOINT_FREQ)

    args = parser.parse_args()
    config = Config.from_args(args)

    from pix2pix import Pix2Pix
    gan = Pix2Pix.new_run(config)
    try:
        gan.train()
    except KeyboardInterrupt:
        print("Stopping...")
