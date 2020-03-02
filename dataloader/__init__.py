from dataloader.facades import Facades
from dataloader.tfrecord import TFRecordDataLoader


DATALOADERS = {
    Facades.__name__: Facades
    TFRecordDataLoader.__name__: TFRecordDataLoader
}
