import pandas as pd
from typing import Union
import json
import os

from yolo_labels.shared import LabelReader


class LabelBoxLabelReader(LabelReader):

    def __init__(self, input_path: str, label_id_mapper: dict, separator: str = ',', images_folder: str = '',
                 ignore_unmapped_labels: bool = True):
        self.input_path = input_path
        self.label_id_mapper = label_id_mapper
        self.separator = separator
        self.images_folder = images_folder
        self.ignore_unmapped_labels = ignore_unmapped_labels
        super(LabelBoxLabelReader, self).__init__(input_path)

    def read_source_file(self, *args, **kwargs) -> pd.DataFrame:
        return pd.read_csv(self.input_path, sep=self.separator, index_col='ID')

    def next_labels(self) -> tuple:
        for index, row in self.data.iterrows():
            objects = json.loads(row['Label'])['objects']
            objects_to_be_yielded = []
            for obj in objects:
                labels = self.extract_object_labels(obj)
                if labels is None:
                    continue
                objects_to_be_yielded += [labels]

            yield os.path.join(self.images_folder, row['External ID']), objects_to_be_yielded

    def extract_object_labels(self, obj) -> Union[tuple, None]:
        label_id = self.get_label_id(obj['title'])
        if label_id is None:
            return None
        bbox = obj['bbox']
        x, y, h, w = bbox['top'], bbox['left'], bbox['height'], bbox['width']
        # example of labels: [('provider', ['evo']),
        # ('parking_place', ['sidewalk', 'reserved_parking_space']), ('is_well_parked', ['yes'])]
        # labels = [self.get_classification_values(cls) for cls in obj['classifications']]
        return x, y, h, w, label_id

    def get_classification_values(self, classification: object) -> tuple:
        title = classification['title']

        if 'answers' in classification:  # self.has_key(classification, 'answers')
            return title, self.get_object_labels(classification['answers'])
        elif 'answer' in classification:
            return title, self.get_object_label(classification['answer'])

        return title, []

    def label_to_id(self, label):
        return self.label_id_mapper[label]

    @classmethod
    def get_object_label(cls, classification: object) -> list:
        return [classification['value']]

    @classmethod
    def get_object_labels(cls, classifications: list) -> list:
        return [cls['value'] for cls in classifications]

    @classmethod
    def has_key(cls, obj: object, key: str):
        try:
            _ = obj[key]
            return True
        except KeyError:
            return False

    def get_label_id(self, label: str):
        try:
            return self.label_to_id(label)
        except KeyError as e:
            self.on_label_map_key_error(e)

    def on_label_map_key_error(self, e: KeyError):
        if self.ignore_unmapped_labels:
            return None
        else:
            raise e
