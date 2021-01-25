import os

from scripts.shared import create_versioned_dataset_dir
from yolo_labels.oid.reader import OIDLabelReader, DatasetType
from yolo_labels.shared import LabelWriter, VehicleDetectionObjectNameId, AnnotationConcatenator
from yolo_labels.vott.reader import VoTTLabelReader
from yolo_labels.yolo_v3_to_v5 import V3ToV5Converter, YoloV5DatasetType

# creating necessary folders

os.makedirs('dist/vehicle_detection', exist_ok=True)
final_dir_path_formatted = '/Users/bothmena/Projects/datasets/vehicles_object_detection/yolo_v5_{}'
final_dir_path = create_versioned_dataset_dir(final_dir_path_formatted)

# Generating YOLOv4 from VoTT

vott_directory = '/Users/bothmena/Projects/datasets/vott-csv-export'
vott_input_path = os.path.join(vott_directory, 'annotations/Annotations-export.csv')
assert os.path.isfile(vott_input_path), 'vott input file does not exist'

vott_reader = VoTTLabelReader(input_path=vott_input_path,
                              label_id_mapper={'escooter': VehicleDetectionObjectNameId.E_SCOOTER.value},
                              object_labels='escooter',
                              images_folder=vott_directory,
                              normalize_bboxes=True,
                              ignore_unmapped_labels=True)
vott_writer = LabelWriter(output_path='dist/vehicle_detection/output_vott.txt',
                          reader=vott_reader, overwrite_existent=True)
vott_writer.write_annotations()

# Generating YOLOv4 from OID

oid_input_path = '/Users/bothmena/Projects/datasets/OID'
oid_label_id_mapper = {
    '/m/0199g': VehicleDetectionObjectNameId.BICYCLE.value,
    '/m/0k4j': VehicleDetectionObjectNameId.CAR.value,
    '/m/04_sv': VehicleDetectionObjectNameId.MOTORCYCLE.value,
    '/m/076bq': VehicleDetectionObjectNameId.SEGWAY.value,
    '/m/01jfm_': VehicleDetectionObjectNameId.LICENCE_PLATE.value,
}

for dataset_type in [DatasetType.TRAIN, DatasetType.VALIDATION]:
    reader = OIDLabelReader(input_path=oid_input_path,
                            dataset_type=dataset_type,
                            label_id_mapper=oid_label_id_mapper,
                            unnormalize_bbox=False,
                            )
    writer = LabelWriter(output_path='dist/vehicle_detection/output_oid_{}.txt'.format(dataset_type.value),
                         reader=reader, overwrite_existent=True)
    writer.write_annotations()

# Concatenating VoTT & OID annotations

output_all = 'dist/vehicle_detection/output_all.txt'
concatenator = AnnotationConcatenator(
    'dist/vehicle_detection/output_vott.txt',
    'dist/vehicle_detection/output_oid_train.txt',
    'dist/vehicle_detection/output_oid_validation.txt',
    output_file=output_all,
)
concatenator()

# Converting annotation from YOLO v4 -> v5

images_dir = '/Users/bothmena/Projects/datasets/vehicles_object_detection/yolo_v3/images/train'
sizes = {
    YoloV5DatasetType.TRAIN: 0.8,
    YoloV5DatasetType.VALIDATION: 0.2,
}

converter = V3ToV5Converter(input_file=output_all,
                            images_dir=images_dir,
                            output_dir=final_dir_path,
                            sizes=sizes)
converter()
converter.write_data_yaml_file(VehicleDetectionObjectNameId)
