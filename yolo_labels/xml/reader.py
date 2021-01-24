import xml.etree.ElementTree as ET
import os
from typing import List, Tuple, Union

from yolo_labels.shared import LabelReader


coord_indices = {
    'xmin': 0,
    'ymin': 1,
    'xmax': 2,
    'ymax': 3
}


class XMLLabelReader(LabelReader):

    def __init__(self,
                 input_path: str,  # dataset folder: should have images_dir & annotations_dir sub-dirs
                 label_id_mapper: dict,
                 annotations_dir: str = 'annotations',
                 images_dir: str = 'images',
                 ignore_unmapped_labels: bool = True
                 ):
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        self.root = None
        super().__init__(input_path, label_id_mapper, ignore_unmapped_labels)

    def next_labels(self) -> tuple:
        for annotations_filename in self.list_annotation_files():
            self.read_source_file(annotations_filename)
            image_filename = self.get_labeled_image_filename()
            labels = self.get_labels()

            yield image_filename, labels

    def list_annotation_files(self):
        all_files = os.listdir(os.path.join(self.input_path, self.annotations_dir))
        return [f for f in all_files if os.path.splitext(f)[1] == '.xml']

    def read_source_file(self, annotations_filename: str):
        annotations_path = os.path.join(self.input_path, self.annotations_dir, annotations_filename)
        parsed_xml = ET.parse(annotations_path)
        self.root = parsed_xml.getroot()

    def get_labeled_image_filename(self) -> str:
        return self.root.find('filename').text

    def get_labels(self) -> List[Tuple[int]]:
        bboxes = []

        for obj in self.root.iter('object'):
            bbox = [0, 0, 0, 0, self.get_label_id(obj.find('name').text)]
            for bndbox in obj.iter('bndbox'):
                for coord in bndbox.findall('*'):
                    bbox[coord_indices[coord.tag]] = int(coord.text)
            bboxes.append(tuple(bbox))

        return bboxes
