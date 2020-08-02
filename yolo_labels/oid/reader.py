import os
from enum import Enum

import cv2
import pandas as pd

from yolo_labels.shared import LabelReader


class DatasetType(Enum):
    TEST = 'test'
    VALIDATION = 'validation'
    TRAIN = 'train'


CSV_PATH = 'csv_folder'
IMAGES_FORMATTED_PATH = 'Dataset/{set}/{label}'
CLASS_DESCRIPTION_PATH = 'class-descriptions-boxable.csv'


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
                 class_desc_filename: str = CLASS_DESCRIPTION_PATH,
                 ignore_unmapped_labels: bool = True,
                 ):
        super(OIDLabelReader).__init__(input_path, label_id_mapper, ignore_unmapped_labels)
        self.dataset_type = dataset_type
        self.csv_path = csv_path
        self.images_formatted_path = images_formatted_path
        self.class_desc_filename = class_desc_filename
        self.label_names = self.get_label_names_dictionary()

    def read_source_file(self, dataset: str, **kwargs) -> pd.DataFrame:
        path = os.path.join(self.input_path, self.csv_path, '{dataset}-annotations-bbox.csv'
                            .format(dataset=self.dataset_type))

        return pd.read_csv(path)

    def read_class_description(self) -> pd.DataFrame:
        path = os.path.join(self.input_path,
                            self.csv_path,
                            self.class_desc_filename)

        return pd.read_csv(path, header=None, names=['name'])

    def get_label_names_dictionary(self) -> dict:
        output = {}
        description_df = self.read_class_description()

        for index, row in description_df.iterrows():
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
        # total_images = oid_annotation.shape[0]
        # test_images = 1002 + 41 + 41 + 5 + 106
        # found_images = 0
        # read_images = 0
        for index, row in self.data.iterrows():
            # read_images += 1
            label_name = self.label_names[row['LabelName']]

            if label_name not in ['Bicycle', 'Car', 'Motorcycle', 'Segway', 'Vehicle registration plate']:
                continue

            img_filename = row['ImageID'] + '.png'
            image_path = self.get_absolute_image_path(label_name, img_filename)

            # found_images += os.path.isfile(image_path)

            # print('rows: {}/{} || found images: {}/{} || Progress: {:.1f}%'.format(read_images, total_images,
            #                                                                        found_images, test_images,
            #                                                                        100 * read_images / total_images),
            #       flush=True)

            if not os.path.isfile(image_path):
                continue

            normalized_bbox_attributes = row['XMin'], row['YMin'], row['XMax'], row['YMax']
            x_min, y_min, x_max, y_max = self.get_bbox_unnormalized_attributes(image_path, normalized_bbox_attributes)

            yield img_filename, x_min, y_min, x_max, y_max, label_name
