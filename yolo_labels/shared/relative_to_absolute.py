import io
from os.path import join


class RelativeToAbsolute:
    def __init__(
            self,
            input_file: str,
            output_file: str,
            absolute_path: str,
    ):
        self.input_filename = input_file
        self.output_file = output_file
        self.absolute_path = absolute_path
        self.input_file = None

    def __call__(self, *args, **kwargs):
        self.main()

    def main(self):
        input_file = io.open(self.input_filename, 'r')
        with io.open(self.output_file, 'w+') as output_file:
            self.append_abs_path(input_file, output_file)
        input_file.close()

    def append_abs_path(self, input_file, output_file):
        for line in input_file.readlines():
            rel_path, *annotations = line.split(' ')
            new_line_parts = [join(self.absolute_path, rel_path)] + annotations
            output_file.write(' '.join(new_line_parts))
