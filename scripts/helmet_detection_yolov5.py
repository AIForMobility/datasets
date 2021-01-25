import os

from yolo_labels.shared import LabelWriter
from yolo_labels.yolo_v3_to_v5 import V3ToV5Converter, YoloV5DatasetType
from yolo_labels.xml.reader import XMLLabelReader
from scripts.shared import create_versioned_dataset_dir
from yolo_labels.shared import HelmetObjectNameId

# creating necessary folders

os.makedirs('dist/helmet_detection', exist_ok=True)
dataset_directory = '/mnt/c/Users/aymen/Projects/Datasets/HelmetDetection'
final_dir_path_formatted = os.path.join(dataset_directory, 'yolo_v5_{}')
final_dir_path = create_versioned_dataset_dir(final_dir_path_formatted)
label_ids = {
    'Without Helmet': 0,
    'With Helmet': 1,
}

# Generating YOLOv4 from VoTT

reader = XMLLabelReader(input_path=dataset_directory,
                         label_id_mapper=label_ids,
                         ignore_unmapped_labels=True)

output_path = 'dist/helmet_detection/output_xml.txt'
writer = LabelWriter(output_path=output_path,
                          reader=reader, overwrite_existent=True)
writer.write_annotations()

# Converting annotation from YOLO v3 -> v5

images_dir = os.path.join(dataset_directory, 'images')
sizes = {
    YoloV5DatasetType.TRAIN: 0.8,
    YoloV5DatasetType.VALIDATION: 0.2,
}

converter = V3ToV5Converter(input_file=output_path,
                            images_dir=images_dir,
                            output_dir=final_dir_path,
                            sizes=sizes,
                            normalize_bboxes=True)
converter()
converter.write_data_yaml_file(HelmetObjectNameId)
