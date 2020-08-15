from yolo_labels.yolo_v3_to_v5 import V3ToV5Converter, YoloV5DatasetType


input_file = 'yolo_labels/yolo_v3_to_v5/input.txt'
output_dir = 'dist/yolo_v5/'
images_dir = 'yolo_labels/yolo_v3_to_v5/sample/input_images'

converter = V3ToV5Converter(input_file, images_dir, output_dir, YoloV5DatasetType.TRAIN)
converter()
