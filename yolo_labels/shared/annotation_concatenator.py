import io


class AnnotationConcatenator:
    def __init__(
            self,
            *input_files: str,
            output_file: str,
    ):
        self.input_filenames = input_files
        self.output_file = output_file

    def __call__(self, *args, **kwargs):
        self.main()

    def main(self):
        output_file = io.open(self.output_file, 'w+')
        for input_filename in self.input_filenames:
            self.append_file_to_output_file(input_filename, output_file)
        output_file.close()

    def append_file_to_output_file(self, input_filename: str, output_file):
        with io.open(input_filename, 'r') as input_file:
            file_content = input_file.read()
            output_file.write(file_content)
