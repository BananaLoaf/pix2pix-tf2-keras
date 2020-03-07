from argparse import ArgumentParser
import pickle

from pathlib import Path
import tensorflow as tf
import cv2
import numpy as np


OUTPUT = "OUTPUT"
RES = "RES"
GS_A = "GS_A"
GS_B = "GS_B"
PATH_A = "PATH_A"
PATH_B = "PATH_B"


def imread(path: Path, gs: bool, resolution: int) -> np.ndarray:
    mode = cv2.IMREAD_GRAYSCALE if gs else cv2.IMREAD_COLOR
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
    parser.add_argument("--A-gs", action="store_true", default=False, help="A is grayscale (default: %(default)s)",
                        dest=GS_A)
    parser.add_argument("--B-gs", action="store_true", default=False, help="B is grayscale (default: %(default)s)",
                        dest=GS_B)
    parser.add_argument(PATH_A, type=str, help="Path to A")
    parser.add_argument(PATH_B, type=str, help="Path to B")
    args = vars(parser.parse_args())

    if not args[OUTPUT].endswith(".tfrecord"):
        args[OUTPUT] += ".tfrecord"
    path_A = Path(args[PATH_A])
    path_B = Path(args[PATH_B])

    with tf.io.TFRecordWriter(args[OUTPUT]) as writer:
        for i, filename_A in enumerate(path_A.glob("*.*")):
            try:
                filename_B = next(path_B.glob(f"{filename_A.stem}.*"))
                print(f"{i}) A: {filename_A}, B: {filename_B}")

                img_A = imread(path=filename_A, gs=args[GS_A], resolution=args[RES])
                img_B = imread(path=filename_B, gs=args[GS_B], resolution=args[RES])

                pair_dict = {
                    "i": tf.train.Feature(int64_list=
                                          tf.train.Int64List(value=[i])
                                          ),
                    "A": tf.train.Feature(bytes_list=
                                          tf.train.BytesList(value=[pickle.dumps(img_A)])
                                          ),
                    "B": tf.train.Feature(bytes_list=
                                          tf.train.BytesList(value=[pickle.dumps(img_B)])
                                          ),
                }
                pair = tf.train.Features(feature=pair_dict)

                example = tf.train.Example(features=pair)
                writer.write(example.SerializeToString())

            except StopIteration:
                print(f"{i}) A: {filename_A}, B: -")
