import pandas as pd

from yolo_labels.shared import LabelReader


class LabelBoxLabelReader(LabelReader):
    def read_source_file(self, *args, **kwargs) -> pd.DataFrame:
        pass

    def next_labels(self) -> tuple:
        pass
