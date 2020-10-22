from yolo_labels.yolo_v3_to_v5 import V3ToV5Converter, YoloV5DatasetType


input_file = 'yolo_labels/yolo_v3_to_v5/input.txt'
output_dir = 'dist/yolo_v5/'
images_dir = 'yolo_labels/yolo_v3_to_v5/sample/input_images'

sizes = {
    YoloV5DatasetType.TRAIN: 0.7,
    YoloV5DatasetType.VALIDATION: 0.3,
}

converter = V3ToV5Converter(input_file, images_dir, output_dir, sizes=sizes)
converter()
