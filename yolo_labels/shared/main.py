from yolo_labels.shared.annotation_concatenator import AnnotationConcatenator

concatenator = AnnotationConcatenator(
    'dist/output.txt',
    'dist/output1.txt',
    'dist/output2.txt',
    output_file='dist/output_all.txt',
)
concatenator()
