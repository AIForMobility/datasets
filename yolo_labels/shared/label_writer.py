import os

from .label_reader import LabelReader


class LabelWriter:
    def __init__(self,
                 output_path: str,
                 reader: LabelReader,
                 overwrite_existent: bool = False):

        self.output_path = output_path
        self.reader = reader
        self.overwrite_existent = overwrite_existent
        self.file = None

    def create_output_file(self):
        if not self.overwrite_existent and os.path.isfile(self.output_path):
            raise FileExistsError('Output file already exist, please change output path or set overwrite_existent to '
                                  'True if you want to over write the old file')

        return open(self.output_path, 'w+')

    def write_annotations(self) -> None:
        """
        loop through all image paths provided by the reader instance and write all annotations to the output file.
        :return:
        """

        self.file = self.create_output_file()

        for (image_path, objects) in self.reader.next_labels():
            self.insert_line(image_path, objects)

        self.on_exit()

    def on_exit(self) -> None:
        self.file.close()
        self.file = None

    def insert_line(self, image_path: str, bboxes: list):
        print('image path', image_path)
        print('objects: ', bboxes)
        print('annotations: ', self.get_image_annotations(image_path, bboxes))
        print('-' * 100)

    def get_image_annotations(self, image_path: str, bboxes: list) -> str:
        return '%s %s' % (image_path, self.bboxes_to_str(bboxes))

    def bboxes_to_str(self, bboxes: list) -> str:
        bboxes_str = ''
        for bbox in bboxes:
            bboxes_str += self.bbox_to_str(bbox)

        return bboxes_str

    @staticmethod
    def bbox_to_str(bbox: tuple) -> str:
        return '%f,%f,%f,%f,%d' % bbox
