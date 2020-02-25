import sys
import inspect

from generator.unet import *


GENERATORS = {
    UNet64.__name__: UNet64,
    UNet128.__name__: UNet128,
    UNet256.__name__: UNet256
}
