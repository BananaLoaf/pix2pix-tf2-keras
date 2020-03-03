import sys
import inspect

from generator.unet import *


GENERATORS = {
    UNet32.__name__: UNet32,
    UNet64.__name__: UNet64,
    UNet128.__name__: UNet128,
    UNet256.__name__: UNet256,
    UNet512.__name__: UNet512,
    UNet1024.__name__: UNet1024
}
