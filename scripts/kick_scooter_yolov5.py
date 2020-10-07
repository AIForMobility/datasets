import os

from yolo_labels.oid.reader import OIDLabelReader, DatasetType
from yolo_labels.shared import LabelWriter, ObjectNameId, AnnotationConcatenator, RelativeToAbsolute
from yolo_labels.vott.reader import VoTTLabelReader

# Generating YOLOv4 from VoTT

vott_directory = '/Users/bothmena/Projects/datasets/vott-csv-export'
vott_input_path = os.path.join(vott_directory, 'annotations/Annotations-export.csv')
assert os.path.isfile(vott_input_path), 'vott input file does not exist'

vott_reader = VoTTLabelReader(input_path=vott_input_path,
                              label_id_mapper={'escooter': ObjectNameId.E_SCOOTER.value},
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
    '/m/0199g': ObjectNameId.BICYCLE.value,
    '/m/0k4j': ObjectNameId.CAR.value,
    '/m/04_sv': ObjectNameId.MOTORCYCLE.value,
    '/m/076bq': ObjectNameId.SEGWAY.value,
    '/m/01jfm_': ObjectNameId.LICENCE_PLATE.value,
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

concatenator = AnnotationConcatenator(
    'dist/output_abs.txt',
    'dist/output1.txt',
    'dist/output2.txt',
    output_file='dist/output_all.txt',
)
concatenator()

rel_to_abs = RelativeToAbsolute('dist/output.txt', 'dist/output_abs.txt', '/Users/bothmena/Projects')
rel_to_abs()

# Converting annotation from YOLO v4 -> v5
