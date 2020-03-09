from argparse import ArgumentParser

from etc.config import Config, CFields
from generator import *
from dataloader import *


ALL_DEVICES = [dev.name for dev in tf.config.list_logical_devices()]


if __name__ == '__main__':
    parser = ArgumentParser(description="Pix2Pix tensorflow 2 keras implementation")
    parser.add_argument("--name", type=str, required=True, help="Model name",
                        dest=CFields.NAME)
    # Model params
    parser.add_argument("-r", "--res", type=int, default=256, help="Input and output image resolution (default: %(default)s)",
                        dest=CFields.RESOLUTION)
    parser.add_argument("--in-channels", type=int, default=3, help="Input image channels (default: %(default)s)",
                        dest=CFields.IN_CHANNELS)
    parser.add_argument("--out-channels", type=int, default=3, help="Output image channels (default: %(default)s)",
                        dest=CFields.OUT_CHANNELS)
    parser.add_argument("-f", "--filters", type=int, default=64, help="Generator filters (default: %(default)s)",
                        dest=CFields.FILTERS)
    parser.add_argument("-g", "--generator", type=str, default=UNet256.__name__, choices=list(GENERATORS.keys()), help="Generator (default: %(default)s)",
                        dest=CFields.GENERATOR)
    parser.add_argument("-d", "--dataloader", type=str, required=True, choices=list(DATALOADERS.keys()), help="DataLoader",
                        dest=CFields.DATALOADER)
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset",
                        dest=CFields.DATASET)
    # Device
    device_group = parser.add_mutually_exclusive_group(required=False)
    device_group.add_argument("--cpu", type=str, default="/device:CPU:0", choices=list(filter(lambda dev: "CPU" in dev, ALL_DEVICES)), help="Single CPU (default: %(default)s)",
                              dest=CFields.DEVICE)
    device_group.add_argument("--gpu", type=str, help=f"Available GPUs: {list(filter(lambda dev: 'GPU' in dev, ALL_DEVICES))}, list devices with , delimiter",
                              dest=CFields.DEVICE)
    device_group.add_argument("--tpu", type=str, help=f"TPU name",
                              dest=CFields.DEVICE)
    parser.add_argument("--xla-jit", action="store_true", default=False, help="XLA Just In Time compilation, does not fully support UpSampling2D layer in the TF 2.1.0 and my never will, https://www.tensorflow.org/xla (default: %(default)s)",
                        dest=CFields.XLA_JIT)
    # Training params
    parser.add_argument("--lr", type=float, default=0.0002, help="Adam Learning rate (default: %(default)s)",
                        dest=CFields.LEARNING_RATE)
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam Beta1 (default: %(default)s)",
                        dest=CFields.BETA1)
    parser.add_argument("--G_L1-lambda", type=float, default=50.0, help="G_L1 lambda (default: %(default)s)",
                        dest=CFields.G_L1_LAMBDA)
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="Batch size (default: %(default)s)",
                        dest=CFields.BATCH_SIZE)
    parser.add_argument("-i", "--iterations", type=int, default=1_000_000, help="Iterations (default: %(default)s)",
                        dest=CFields.ITERATIONS)
    # Training options
    parser.add_argument("-o", "--output", type=str, default="result", help="Output path for run folder (default: %(default)s)",
                        dest=CFields.OUTPUT_PATH)
    parser.add_argument("-sn", "--sample-n", type=int, default=6, help="Amount of samples (default: %(default)s)",
                        dest=CFields.SAMPLE_N)
    parser.add_argument("-sf", "--sample-freq", type=int, default=100, help="Sampling frequency in iterations (default: %(default)s)",
                        dest=CFields.SAMPLE_FREQ)
    parser.add_argument("-cf", "--checkpoint-freq", type=int, default=10_000, help="Checkpoint frequency in iterations (default: %(default)s)",
                        dest=CFields.CHECKPOINT_FREQ)

    args = parser.parse_args()
    config = Config.from_args(args)

    from pix2pix import Pix2Pix
    gan = Pix2Pix.new_run(config)
    try:
        gan.train()
    except KeyboardInterrupt:
        print("Stopping...")
