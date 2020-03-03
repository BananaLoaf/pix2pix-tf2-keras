from argparse import ArgumentParser

import tensorflow as tf

from config import *
from generator import *
from dataloader import *


ALL_DEVICES = [dev.name.replace("device:", "").lower() for dev in tf.config.list_logical_devices()]


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
    # Device
    device_group = parser.add_mutually_exclusive_group(required=False)
    device_group.add_argument("--cpu", type=str, default="/cpu:0", choices=list(filter(lambda dev: dev.startswith("/cpu"), ALL_DEVICES)), help="Single CPU (default: %(default)s)",
                              dest=DEVICE)
    device_group.add_argument("--gpu", type=str, help=f"Available GPUs: {list(filter(lambda dev: dev.startswith('/gpu'), ALL_DEVICES))}, list devices with ; delimiter",
                              dest=DEVICE)
    parser.add_argument("--tpu", action="store_true", default=False, help=f"Experimental utilization of Google Cloud TPUs. If supplied, --cpu and --gpu arguments are ignored",
                        dest=USE_TPU)
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
