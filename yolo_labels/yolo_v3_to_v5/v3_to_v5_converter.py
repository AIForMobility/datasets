import io
import os
import random
import shutil
from typing import List, Dict, Type
from enum import Enum

from PIL import Image

from yolo_labels.yolo_v3_to_v5 import YoloV5DatasetType


class V3ToV5Converter:
    def __init__(
            self,
            input_file: str,
            images_dir: str,
            output_dir: str,
            sizes: Dict[YoloV5DatasetType, float],
            normalize_bboxes: bool = False,
    ):
        self.input_file = input_file
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.sizes = sizes
        self.normalize_bboxes = normalize_bboxes

    def __call__(self):
        self.create_dirs(list(self.sizes.keys()))
        with io.open(self.input_file, 'r') as input_file:
            lines = self.split_lines(input_file.readlines())
            for dataset_type, lines_part in lines.items():
                self.write_dataset_type_annotations(dataset_type, lines_part)

    def write_dataset_type_annotations(self, dataset_type: YoloV5DatasetType, lines: List[str]):
        labels_dir = os.path.join(self.output_dir, dataset_type.value, 'labels')
        images_dir = os.path.join(self.output_dir, dataset_type.value, 'images')
        for line in lines:
            self.write_v5_annotations(line, labels_dir, images_dir)

    def write_v5_annotations(self, line: str, labels_dir: str, output_images_dir: str):
        image, *objects = line.split(' ')
        annotations = self.generate_annotations(image, objects)
        output_filename = '{}.txt'.format(os.path.splitext(image)[0])
        with io.open(os.path.join(labels_dir, output_filename), 'w+') as output_file:
            output_file.write(annotations)
        self.copy_image_file(output_images_dir, image)

    def generate_annotations(self, image, objects: list):
        annotations = ''
        for obj in objects:  # type: str
            obj = obj.strip()
            if obj == '':
                continue
            annotations += self.stringify_object_data(image, obj)

        return annotations

    def stringify_object_data(self, image, obj: str) -> str:
        *bbox, label_id = obj.strip().split(',')
        normalized_bbox = self.normalize_bbox(image, bbox)
        annotation_list = [label_id] + normalized_bbox
        annotation_list = [str(item) for item in annotation_list]

        return ' '.join(annotation_list) + '\n'

    def normalize_bbox(self, image_filename: str, bbox: list):
        float_bbox = [float(item) for item in bbox]

        if not self.normalize_bboxes:
            return float_bbox

        image = Image.open(os.path.join(self.images_dir, image_filename))  # type: Image.Image
        height, width = image.size
        return [float_bbox[0] / width, float_bbox[1] / height, float_bbox[2] / width, float_bbox[3] / height]

    def create_dirs(self, dataset_types: List[YoloV5DatasetType]):
        for dataset_type in dataset_types:
            os.mkdir(os.path.join(self.output_dir, dataset_type.value))
            os.mkdir(os.path.join(self.output_dir, dataset_type.value, 'images'))
            os.mkdir(os.path.join(self.output_dir, dataset_type.value, 'labels'))

    def copy_image_file(self, output_image_dir: str, image: str):
        shutil.copy2(
            os.path.join(self.images_dir, image),
            os.path.join(output_image_dir, image)
        )

    def split_lines(self, lines: List[str]) -> Dict[YoloV5DatasetType, List[str]]:
        random.shuffle(lines)
        lines_length = len(lines)
        sum_sizes = 0
        split_lines = {}
        for dataset_type, size in self.sizes.items():
            start_index = int(sum_sizes * lines_length)
            sum_sizes = float(format(sum_sizes + size, '.5g'))
            end_index = lines_length if sum_sizes == 1 else int(sum_sizes * lines_length)
            split_lines[dataset_type] = lines[start_index:end_index]

        return split_lines

    def write_data_yaml_file(self, labels: Type[Enum]):
        classes_names = [cls.name for cls in labels]
        with open(os.path.join(self.output_dir, 'data.yaml'), 'w') as f:
            f.write('train: ../train/images\n')
            f.write('val: ../valid/images\n')
            f.write('\n')
            f.write('nc: {}\n'.format(len(classes_names)))
            f.write("names: {}".format(classes_names))
