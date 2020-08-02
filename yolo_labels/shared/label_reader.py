from abc import ABCMeta, abstractmethod
import pandas as pd


class LabelReader(metaclass=ABCMeta):
    def __init__(self, input_path: str, label_id_mapper: dict, ignore_unmapped_labels: bool = True):
        super(LabelReader, self).__init__()

        self.input_path = input_path
        self.label_id_mapper = label_id_mapper
        self.ignore_unmapped_labels = ignore_unmapped_labels

    def get_label_id(self, label: str):
        try:
            return self.__label_to_id(label)
        except KeyError as e:
            self.__on_label_map_key_error(e)

    def __label_to_id(self, label):
        return self.label_id_mapper[label]

    def __on_label_map_key_error(self, e: KeyError):
        if self.ignore_unmapped_labels:
            return None
        else:
            raise e

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
