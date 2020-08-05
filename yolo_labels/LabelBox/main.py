from yolo_labels.LabelBox.reader import LabelBoxLabelReader
from yolo_labels.shared import LabelWriter
import os


filename = 'yolo_labels/LabelBox/input.csv'
input_path = os.path.join(os.getcwd(), filename)
assert os.path.isfile(input_path)

reader = LabelBoxLabelReader(input_path, {'moped_bbox': 0}, ignore_unmapped_labels=True)
writer = LabelWriter(output_path='dist/output.txt', reader=reader, overwrite_existent=True)

writer.write_annotations()
