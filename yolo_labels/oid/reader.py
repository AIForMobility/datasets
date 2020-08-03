import os
from enum import Enum

import cv2
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple

from yolo_labels.shared import LabelReader


class DatasetType(Enum):
    TEST = 'test'
    VALIDATION = 'validation'
    TRAIN = 'train'


CSV_PATH = 'csv_folder'
IMAGES_FORMATTED_PATH = 'Dataset/{set}/{label}'
CLASS_DESCRIPTION_FILENAME = 'class-descriptions-boxable.csv'


# input_path:
# path to the folder that contains the annotations & images, the structure of the folder is like this:
# input_path/
# -- csv_folder/
# -- -- class-descriptions-boxable.csv
# -- -- test-annotations-bbox.csv
# -- -- train-annotations-bbox.csv
# -- -- validation-annotations-bbox.csv
# -- Dataset/
# -- -- {test|train|validation}/
# -- -- -- ClassNameX/
# -- -- -- -- image_id.jpg ...
class OIDLabelReader(LabelReader):
    def __init__(self,
                 input_path: str,
                 dataset_type: DatasetType,
                 label_id_mapper: dict,
                 csv_path: str = CSV_PATH,
                 images_formatted_path: str = IMAGES_FORMATTED_PATH,
                 class_desc_filename: str = CLASS_DESCRIPTION_FILENAME,
                 ignore_unmapped_labels: bool = True,
                 ):
        super(OIDLabelReader, self).__init__(input_path=input_path,
                                             label_id_mapper=label_id_mapper,
                                             ignore_unmapped_labels=ignore_unmapped_labels)
        self.dataset_type = dataset_type.value
        self.csv_path = csv_path
        self.images_formatted_path = images_formatted_path
        self.class_desc_filename = class_desc_filename
        self.data = self.process_df()
        self.label_names = self.get_label_names_dictionary()

    def read_source_file(self, dataset: DatasetType, **kwargs) -> pd.DataFrame:
        path = os.path.join(self.input_path, self.csv_path, '{dataset}-annotations-bbox.csv'
                            .format(dataset=self.dataset_type))

        return pd.read_csv(path)\
            .set_index('ImageID', append=True)\
            .swaplevel(0, 1)

    def process_df(self):
        df = self.read_source_file(self.dataset_type)
        df = self.get_filtered_df(df)
        return df

    def get_filtered_df(self, df: pd.DataFrame):
        """
        removes row with labelName that is not part of the labels we're detecting.
        :param df: DataFrame source
        :return: DataFrame
        """
        return df[df['LabelName'].isin(self.label_id_mapper.keys())]

    # def group_by_image_name(self, df: pd.DataFrame):

    def read_class_description(self) -> pd.DataFrame:
        path = os.path.join(self.input_path,
                            self.csv_path,
                            self.class_desc_filename)

        return pd.read_csv(path, header=None, names=['name'])

    def get_label_names_dictionary(self) -> dict:
        output = {}
        description_df = self.read_class_description()

        for index, row in description_df.iterrows():
            if index in self.label_id_mapper.keys():
                output[index] = row['name']

        return output

    def get_absolute_image_path(self, label: str, filename: str) -> str:
        return os.path.join(self.input_path,
                            self.images_formatted_path.format(set=self.dataset_type, label=label),
                            filename)

    @staticmethod
    def get_image_shape(path: str) -> tuple:
        image = cv2.imread(path)
        return image.shape

    def get_bbox_unnormalized_attributes(self, image_path, normalized_bbox_attributes: tuple) -> tuple:
        img_height, img_width, _ = self.get_image_shape(image_path)
        x_min, y_min, x_max, y_max = normalized_bbox_attributes

        return x_min * img_width, y_min * img_height, x_max * img_width, y_max * img_height

    def next_labels(self) -> tuple:
        with tqdm(self.data) as p_bar:
            for image, image_objects in self.data.groupby(level=0):
                p_bar.update()

                img_filename = image + '.jpg'
                label_name = self.label_names[image_objects.iloc[(0, 1)]]
                image_path = self.get_absolute_image_path(label_name, img_filename)
                if not os.path.isfile(image_path):
                    continue

                bboxes = self.get_image_bboxes(image_objects, image_path)

                yield image_path, bboxes

    def get_image_bboxes(self, image_objects: pd.DataFrame, image_path: str) -> List[Tuple]:
        bboxes = []
        for _, row in image_objects.iterrows():
            label_id = self.label_id_mapper[row['LabelName']].value

            normalized_bbox_attributes = row['XMin'], row['YMin'], row['XMax'], row['YMax']
            x_min, y_min, x_max, y_max = self.get_bbox_unnormalized_attributes(image_path, normalized_bbox_attributes)

            bboxes.append((x_min, y_min, x_max, y_max, label_id))

        return bboxes
