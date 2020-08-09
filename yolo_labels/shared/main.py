import os.path as path

from yolo_labels.shared.annotation_concatenator import AnnotationConcatenator
from yolo_labels.shared.relative_to_absolute import RelativeToAbsolute

oid_ann_path = '/Users/bothmena/Projects/datasets/OID/oid_to_yolo'
oid_train = path.join(oid_ann_path, 'output.txt')
oid_train_abs = path.join(oid_ann_path, 'output_abs.txt')
oid_val = path.join(oid_ann_path, 'output_validation.txt')
oid_val_abs = path.join(oid_ann_path, 'output_validation_abs.txt')
vott_ann_path = '/Users/bothmena/Projects/datasets/vott-csv-export/annotations'
vott_ann = path.join(vott_ann_path, 'train.txt')
vott_ann_abs = path.join(vott_ann_path, 'train_abs.txt')

output_file = '/Users/bothmena/Projects/datasets/vehicles_object_detection/g_dive_train.txt'

rel_to_abs = RelativeToAbsolute(oid_train, oid_train_abs, 'abs_path_in_google_drive_linked_to_google_colab')
rel_to_abs()

rel_to_abs = RelativeToAbsolute(oid_val, oid_val_abs, 'abs_path_in_google_drive_linked_to_google_colab')
rel_to_abs()

rel_to_abs = RelativeToAbsolute(vott_ann, vott_ann_abs, 'abs_path_in_google_drive_linked_to_google_colab')
rel_to_abs()

concatenator = AnnotationConcatenator(
    oid_train_abs,
    oid_val_abs,
    vott_ann_abs,
    output_file=output_file,
)
concatenator()
