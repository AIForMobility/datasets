from abc import ABCMeta, abstractmethod
import pandas as pd


class LabelReader(metaclass=ABCMeta):
    def __init__(self, input_path: str):
        super(LabelReader, self).__init__()

        self.input_path = input_path
        self.data = self.read_source_file()  # type: pd.DataFrame

    @abstractmethod
    def read_source_file(self, *args, **kwargs) -> pd.DataFrame:
        """
        should read the file and return an data frame
        :param args:
        :param kwargs:
        :return:
        """

    @abstractmethod
    def next_labels(self) -> tuple:
        """
        a generator that yields a tuple (described below)

        :return: a tuple containing a string (image path) and an array of tuples. => (image_path, [(xxx), (yyy), ...])
        each tuple should have the format: (x,y,w,h,class_id): bounding box coordinates + object class id
        """
