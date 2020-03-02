from dataloader.facades import FacadesDataLoader
from dataloader.tfrecord import TFRecordDataLoader


DATALOADERS = {
    FacadesDataLoader.__name__: FacadesDataLoader,
    TFRecordDataLoader.__name__: TFRecordDataLoader
}
