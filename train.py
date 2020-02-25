from argparse import ArgumentParser

from pix2pix import *

if __name__ == '__main__':
    parser = ArgumentParser(description="Pix2Pix tensorflow 2 keras implementation")
    parser.add_argument("--name", type=str, required=True, help="Model name")
    # Model params
    parser.add_argument("-r", "--res", type=int, default=256, help="Input and output image resolution (default: %(default)s)")
    parser.add_argument("--in-channels", type=int, default=3, help="Input image channels (default: %(default)s)")
    parser.add_argument("--out-channels", type=int, default=3, help="Output image channels (default: %(default)s)")
    parser.add_argument("-f", "--filters", type=int, default=64, help="Generator filters (default: %(default)s)")
    parser.add_argument("-g", "--generator", type=str, default=UNet256.__name__, choices=list(GENERATORS.keys()), help="Generator (default: %(default)s)")
    parser.add_argument("-d", "--dataloader", type=str, required=True, choices=list(DATALOADERS.keys()), help="DataLoader")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    # Training params
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate (default: %(default)s)")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 (default: %(default)s)")
    parser.add_argument("--lmbd", type=int, default=50, help="G_L1 coefficient (default: %(default)s)")
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="Batch size (default: %(default)s)")
    parser.add_argument("-e", "--epochs", type=int, default=1_000, help="Epochs (default: %(default)s)")

    args = parser.parse_args()

    # Start training
    config = {
        NAME: args.name,

        RESOLUTION: args.res,
        IN_CHANNELS: args.in_channels,
        OUT_CHANNELS: args.out_channels,
        FILTERS: args.filters,
        GENERATOR: args.generator,
        DATALOADER: args.dataloader,
        DATASET: args.dataset,

        LEARNING_RATE: args.lr,
        BETA1: args.beta1,
        LAMBDA: args.lmbd,
        BATCH_SIZE: args.batch_size,
        STEP: 0,
        EPOCH: 0,
        EPOCHS: args.epochs
    }

    gan = Pix2Pix.new_run(config)
    gan.train()
