import pandas as pd
from typing import List, Tuple, Union
import os

from yolo_labels.shared import LabelReader


class VoTTLabelReader(LabelReader):

    def __init__(self,
                 input_path: str,
                 label_id_mapper: dict,
                 object_labels: Union[str, List[str]],
                 separator: str = ',',
                 images_folder: str = '',
                 ignore_unmapped_labels: bool = True):
        self.input_path = input_path
        self.label_id_mapper = label_id_mapper
        self.object_labels = object_labels
        self.separator = separator
        self.images_folder = images_folder
        self.ignore_unmapped_labels = ignore_unmapped_labels
        super(VoTTLabelReader, self).__init__(input_path, label_id_mapper, ignore_unmapped_labels)
        self.data = self.read_source_file()  # type: pd.DataFrame

    def read_source_file(self, *args, **kwargs) -> pd.DataFrame:
        df = pd.read_csv(self.input_path, sep=self.separator)  # , index_col=['label', 'image']
        filtered_df = self.filter_df(df)
        return filtered_df

    def next_labels(self) -> tuple:
        for index, group in self.data.groupby('image'):
            image_path = os.path.join(self.images_folder, group.iloc[0, 0])
            labeled_bboxes = self.get_image_bboxes(group)

            yield image_path, labeled_bboxes

    def get_image_bboxes(self, group: pd.DataFrame) -> List[Tuple[int, int, int, int, int]]:
        bboxes = []
        for _, bbox_row in group.iterrows():
            bbox = self.get_image_bboxes_labels(bbox_row)
            if bbox is None:
                continue
            bboxes += [bbox]

        return bboxes

    def get_image_bboxes_labels(self, row: pd.Series):
        label_id = self.get_label_id(row['label'])
        if label_id is None:
            return None

        return row['xmin'], row['ymin'], row['xmax'], row['ymax'], label_id

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if type(self.object_labels) is str:
            return df[df['label'] == self.object_labels]
        elif type(self.object_labels) is List[str]:
            return df[df['label'].isin(self.object_labels)]
