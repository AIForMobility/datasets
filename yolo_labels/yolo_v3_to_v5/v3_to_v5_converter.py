import io
from os import path
from PIL import Image

from yolo_labels.yolo_v3_to_v5 import YoloV5DatasetType


class V3ToV5Converter:
    def __init__(
            self,
            input_file: str,
            images_dir: str,
            output_dir: str,
            dataset_type: YoloV5DatasetType
    ):
        self.input_file = input_file
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.dataset_type = dataset_type.value

    def __call__(self):
        with io.open(self.input_file, 'r') as input_file:
            for line in input_file.readlines():
                self.write_v5_annotations(line)

    def write_v5_annotations(self, line: str):
        image, *objects = line.split(' ')
        annotations = self.generate_annotations(image, objects)
        output_filename = '{}.txt'.format(path.splitext(image)[0])
        with io.open(path.join(self.output_dir, output_filename)) as output_file:
            output_file.write(annotations)

    def generate_annotations(self, image, objects: list):
        annotations = ''
        for obj in objects:
            annotations += self.stringify_object_data(image, obj)

        return annotations

    def stringify_object_data(self, image, obj) -> str:
        *bbox, label_id = obj
        normalized_bbox = self.normalize_bbox(image, bbox)
        annotation_list = [label_id] + normalized_bbox
        annotation_list = [str(item) for item in annotation_list]

        return ' '.join(annotation_list)

    def normalize_bbox(self, image_filename: str, bbox: list):
        image = Image.open(path.join(self.images_dir, image_filename))  # type: Image.Image
        width, height = image.size
        width, height = float(width), float(height)

        return [bbox[0]/width, bbox[1]/width, bbox[2]/height, bbox[3]/height]
