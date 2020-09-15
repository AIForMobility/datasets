from yolo_labels.vott.reader import VoTTLabelReader
from yolo_labels.shared import LabelWriter, ObjectNameId
import os


filename = 'yolo_labels/vott/input.csv'
input_path = os.path.join(os.getcwd(), filename)
print(input_path)
assert os.path.isfile(input_path)

reader = VoTTLabelReader(input_path=input_path,
                         label_id_mapper={'moped_bbox': ObjectNameId.MOTORCYCLE.value},
                         object_labels='moped_bbox',
                         ignore_unmapped_labels=True)
writer = LabelWriter(output_path='dist/output_vott.txt', reader=reader, overwrite_existent=True)

writer.write_annotations()
