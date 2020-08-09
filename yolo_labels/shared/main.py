from yolo_labels.shared.annotation_concatenator import AnnotationConcatenator
from yolo_labels.shared.relative_to_absolute import RelativeToAbsolute

rel_to_abs = RelativeToAbsolute('dist/output.txt', 'dist/output_abs.txt', '/Users/bothmena/Projects')
rel_to_abs()

concatenator = AnnotationConcatenator(
    'dist/output_abs.txt',
    'dist/output1.txt',
    'dist/output2.txt',
    output_file='dist/output_all.txt',
)
concatenator()
