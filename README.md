# pix2pix-tf2-keras

[![Python Version](https://img.shields.io/badge/Python-%3E%3D3.6-blue)](https://www.python.org/downloads/)
[![Tensorflow Version](https://img.shields.io/badge/Tensorflow-2.1.0-yellow)](https://github.com/tensorflow/tensorflow/tree/v2.1.0)
[![Tensorflow Version](https://img.shields.io/badge/CUDA-%3E%3D10.0-green)](https://developer.nvidia.com/cuda-downloads)
[![GitHub Licence](https://img.shields.io/github/license/BananaLoaf/pix2pix-tf2-keras.svg?color=blue)](https://github.com/BananaLoaf/pix2pix-tf2-keras/blob/master/LICENSE)

**Pix2Pix** made using **Keras** with **TensorFlow 2** as backend

# Getting started

```bash
pip install -r requirements.txt
python train.py --help
python resume.py --help
python dataset2tfrecord.py --help
```

# Dataloader

Convert your dataset into **TFRecord** file with ```dataset2tfrecord.py``` and choose **TFRecordDataLoader** as dataloader. Converter requires two folders in which related images must have the same names (**with any extension!**)

### or

Make your own dataloader using ```dataloader/template.py``` and add it to ```dataloader/__init__.py.DATALOADERS``` dictionary for it to show up in training options

# Resuming training

Model creates separate folders for each run in ```result/``` folder. Each run has its ```config.json```, if you are not sure what you are changing there, don't. To resume training from the last checkpoint pass run path (**not the config path!**) to ```resume.py```

# Custom generators

Make your own generator using ```generator/template.py``` and add it to ```generator/__init__.py.GENERATORS``` dictionary for it to show up in training options

