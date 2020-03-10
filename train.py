from argparse import ArgumentParser

from etc.config import Config, CF
from generator import *
from dataloader import *


ALL_DEVICES = [dev.name for dev in tf.config.list_logical_devices()]
PLOT = "PLOT"
OUTPUT_PATH = "OUTPUT_PATH"

if __name__ == '__main__':
    parser = ArgumentParser(description="Pix2Pix tensorflow 2 keras implementation")
    parser.add_argument("--name", type=str, required=True, help="Model name",
                        dest=CF.NAME)
    # Model params
    parser.add_argument("-r", "--res", type=int, default=256, help="Input and output image resolution (default: %(default)s)",
                        dest=CF.RESOLUTION)
    parser.add_argument("--in-channels", type=int, default=3, help="Input image channels (default: %(default)s)",
                        dest=CF.IN_CHANNELS)
    parser.add_argument("--out-channels", type=int, default=3, help="Output image channels (default: %(default)s)",
                        dest=CF.OUT_CHANNELS)
    parser.add_argument("-f", "--filters", type=int, default=64, help="Generator filters (default: %(default)s)",
                        dest=CF.FILTERS)
    parser.add_argument("-g", "--generator", type=str, default=UNet256.__name__, choices=list(GENERATORS.keys()), help="Generator (default: %(default)s)",
                        dest=CF.GENERATOR)
    parser.add_argument("-d", "--dataloader", type=str, required=True, choices=list(DATALOADERS.keys()), help="DataLoader",
                        dest=CF.DATALOADER)
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset",
                        dest=CF.DATASET)
    parser.add_argument("--plot", action="store_true", help="Plot networks into png, requires pydot and graphviz installed",
                        dest=PLOT)
    # Device
    device_group = parser.add_mutually_exclusive_group(required=False)
    device_group.add_argument("--cpu", type=str, default="/device:CPU:0", choices=list(filter(lambda dev: "CPU" in dev, ALL_DEVICES)), help="Single CPU (default: %(default)s)",
                              dest=CF.DEVICE)
    device_group.add_argument("--gpu", type=str, help=f"Available GPUs: {list(filter(lambda dev: 'GPU' in dev, ALL_DEVICES))}, list devices with , delimiter",
                              dest=CF.DEVICE)
    device_group.add_argument("--tpu", type=str, help=f"TPU name",
                              dest=CF.DEVICE)
    parser.add_argument("--xla-jit", action="store_true", default=False, help="XLA Just In Time compilation, does not fully support UpSampling2D layer in the TF 2.1.0 and my never will, https://www.tensorflow.org/xla (default: %(default)s)",
                        dest=CF.XLA_JIT)
    # Training params
    parser.add_argument("--lr", type=float, default=0.0002, help="Adam Learning rate (default: %(default)s)",
                        dest=CF.LEARNING_RATE)
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam Beta1 (default: %(default)s)",
                        dest=CF.BETA1)
    parser.add_argument("--G_L1-lambda", type=float, default=50.0, help="G_L1 lambda (default: %(default)s)",
                        dest=CF.G_L1_LAMBDA)
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="Batch size (default: %(default)s)",
                        dest=CF.BATCH_SIZE)
    parser.add_argument("-i", "--iterations", type=int, default=1_000_000, help="Iterations (default: %(default)s)",
                        dest=CF.ITERATIONS)
    # Training options
    parser.add_argument("-o", "--output", type=str, default="result", help="Output path for run folder (default: %(default)s)",
                        dest=OUTPUT_PATH)
    parser.add_argument("-sn", "--sample-n", type=int, default=6, help="Amount of samples (default: %(default)s)",
                        dest=CF.SAMPLE_N)
    parser.add_argument("-sf", "--sample-freq", type=int, default=100, help="Sampling frequency in iterations (default: %(default)s)",
                        dest=CF.SAMPLE_FREQ)
    parser.add_argument("-cf", "--checkpoint-freq", type=int, default=10_000, help="Checkpoint frequency in iterations (default: %(default)s)",
                        dest=CF.CHECKPOINT_FREQ)

    args = parser.parse_args()
    config = Config.from_args(args)
    del config[OUTPUT_PATH]
    del config[PLOT]

    from pix2pix import Pix2Pix
    gan = Pix2Pix.new_run(config, output=getattr(args, OUTPUT_PATH), plot=getattr(args, PLOT))
    try:
        gan.train()
    except KeyboardInterrupt:
        print("Stopping...")
