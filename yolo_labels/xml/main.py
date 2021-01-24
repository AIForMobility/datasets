from yolo_labels.xml.reader import XMLLabelReader
from yolo_labels.shared import LabelWriter, ObjectNameId
import os


filename = 'yolo_labels/xml/sample'
input_path = os.path.join(os.getcwd(), filename)
print(input_path)
assert os.path.isdir(input_path)

label_ids = {
    'Without Helmet': 0,
    'With Helmet': 1,
}

reader = XMLLabelReader(input_path=input_path,
                         label_id_mapper=label_ids,
                         ignore_unmapped_labels=True)
writer = LabelWriter(output_path='dist/output_xml.txt', reader=reader, overwrite_existent=True)

writer.write_annotations()
