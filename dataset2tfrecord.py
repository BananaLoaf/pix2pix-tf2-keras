from argparse import ArgumentParser
import pickle

from pathlib import Path
import tensorflow as tf
import cv2
import numpy as np


OUTPUT = "OUTPUT"
RES = "RES"
BnW_A = "BnW_A"
BnW_B = "BnW_B"
PATH_A = "PATH_A"
PATH_B = "PATH_B"


def imread(path: Path, bnw: bool, resolution: int) -> np.ndarray:
    mode = cv2.IMREAD_GRAYSCALE if bnw else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), mode)
    img = cv2.resize(img, (resolution, resolution))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float) / 127.5 - 1


if __name__ == '__main__':
    parser = ArgumentParser(description="TFRecord file generator")
    parser.add_argument("-o", type=str, default="default.tfrecord", help="Output for TFRecord file (default: %(default)s)",
                        dest=OUTPUT)
    parser.add_argument("-r", "--res", type=int, default=256, help="Image resolution (default: %(default)s)",
                        dest=RES)
    parser.add_argument("--A-bnw", action="store_true", default=False, help="A is black and write (default: %(default)s)",
                        dest=BnW_A)
    parser.add_argument("--B-bnw", action="store_true", default=False, help="B is black and write (default: %(default)s)",
                        dest=BnW_B)
    parser.add_argument(PATH_A, type=str, help="Path to A")
    parser.add_argument(PATH_B, type=str, help="Path to B")
    args = vars(parser.parse_args())

    if not args[OUTPUT].endswith(".tfrecord"):
        args[OUTPUT] += ".tfrecord"
    path_A = Path(args[PATH_A])
    path_B = Path(args[PATH_B])

    with tf.io.TFRecordWriter(args[OUTPUT]) as writer:
        for filename_A in path_A.glob("*.*"):
            filename_B = next(path_B.glob(f"{filename_A.stem}.*"))
            print(f"A: {filename_A}, B: {filename_B}")

            img_A = imread(path=filename_A, bnw=args[BnW_A], resolution=args[RES])
            img_B = imread(path=filename_B, bnw=args[BnW_B], resolution=args[RES])

            A_list = tf.train.BytesList(value=[pickle.dumps(img_A)])
            B_list = tf.train.BytesList(value=[pickle.dumps(img_B)])
            pair_dict = {
                "A": tf.train.Feature(bytes_list=A_list),
                "B": tf.train.Feature(bytes_list=B_list),
            }
            pair = tf.train.Features(feature=pair_dict)

            example = tf.train.Example(features=pair)
            writer.write(example.SerializeToString())
