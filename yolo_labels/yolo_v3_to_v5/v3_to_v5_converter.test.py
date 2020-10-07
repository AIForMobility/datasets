# import unittest
from yolo_labels.yolo_v3_to_v5 import YoloV5DatasetType, V3ToV5Converter


class ConverterTestSuit: # (unittest.TestCase):
    def __init__(self):
        super(ConverterTestSuit, self).__init__()
        self.triple_sizes = {
            YoloV5DatasetType.TRAIN: 0.6,
            YoloV5DatasetType.VALIDATION: 0.3,
            YoloV5DatasetType.TEST: 0.1,
        }

    def test_split_lines_sum_equal_1(self):
        lines = ['line_nbr_{}'.format(n) for n in range(1001)]
        converter = V3ToV5Converter('xxx', 'yyy', 'zzz', sizes=self.triple_sizes)

        split_lines = converter.split_lines(lines)
        sum = 0
        for key, lines_part in split_lines.items():
            sum += len(lines_part)

        assert sum == len(lines), 'sum of all lines ({}) should match sum of all split lines ({})'\
            .format(len(lines), sum)
        assert len(split_lines[YoloV5DatasetType.TEST]) == 101, 'sum of test lines ({}) should be == 101'\
            .format(len(split_lines[YoloV5DatasetType.TEST]))


if __name__ == '__main__':
    print('starting test suite for converter...')
    conv_test = ConverterTestSuit()
    conv_test.test_split_lines_sum_equal_1()
