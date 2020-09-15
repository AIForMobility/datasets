from yolo_labels.LabelBox.reader import LabelBoxLabelReader
from yolo_labels.shared import LabelWriter, ObjectNameId
import os


filename = 'yolo_labels/LabelBox/input.csv'
input_path = os.path.join(os.getcwd(), filename)
assert os.path.isfile(input_path)

reader = LabelBoxLabelReader(input_path, {'moped_bbox': ObjectNameId.MOTORCYCLE.value}, ignore_unmapped_labels=True)
writer = LabelWriter(output_path='dist/output_label_box.txt', reader=reader, overwrite_existent=True)

writer.write_annotations()
