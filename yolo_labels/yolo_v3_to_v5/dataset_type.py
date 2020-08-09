from enum import Enum


class YoloV5DatasetType(Enum):
    VALIDATION = 'valid'
    TRAIN = 'train'
    TEST = 'test'
