"""
a class that converts a dataset of images saved as a csv file to image files
the csv file is expected to have two columns:
- client: which is a string: name of the client owning the image
- data: an image encoded in base64

This class will also create a csv file that will list all images in s3 that can be later fed to a Sagemaker batch transform job to run
prediction on all images that will be later (after conversion: b64 -> images) uploaded to S3, for this you need to provide to additional
arguments:
- s3_bucket: bucket name that is storing the images
- s3_dir_key: S3 key of the directory that will contain the uploaded images (under this directory you should find all client's sub-dirs)
example of an image s3 URI: s3://some-s3-bucket/directory-1/directory-2/client/client_000123.jpg
---> s3_bucket = some-s3-bucket
     s3_dir_key = directory-1/directory-2

the script will read the csv file and save all images as a jpg file in the following structure:
`output_dir`
  |- `client`
    |- `{client}_{:06d}.jpg (example: easyway_000123.jpg)
"""
import os
from base64 import b64decode

import pandas as pd
import tqdm


class Base64ToImageFileConverter:

    image_fn_format = '{}_{:06d}.jpg'
    error_fn_format = 'error_{:07d}.txt'

    def __init__(self,
                 csv_path: str,
                 output_dir: str,
                 s3_bucket: str = None,
                 s3_dir_key: str = None):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.s3_bucket = s3_bucket
        self.s3_dir_key = s3_dir_key[1:] if s3_dir_key[0] == '/' else s3_dir_key
        self.counter = {}
        self.unsuccessful_conversions = 0
        self.images_dir = self.create_sub_directory('images')
        self.error_dir = self.create_sub_directory('errors')

    @staticmethod
    def sanitize_base64_str(b64_data: str) -> str:
        """
        some base64 are a binary string in the form "b'__base_64__'
        => so in this case we emit the first 2 and last chars
        """
        if b64_data[:2] == "b'":
            return b64_data[2:-1]
        return b64_data

    def increment_or_init_client_count(self, client: str):
        try:
            self.counter[client] += 1
            return self.counter[client]
        except KeyError:
            self.counter[client] = 1
            return 1

    def get_abs_file_path(self, client: str, filename: str):
        dir_path = os.path.join(self.images_dir, client)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        return os.path.join(dir_path, filename)

    def save_image(self, data: str, client: str, filename: str):
        img_data = b64decode(data)
        file_path = self.get_abs_file_path(client, filename)
        with open(file_path, 'wb') as f:
            f.write(img_data)

    def write_image_file(self, row: pd.Series):
        client = row['client']
        b64_image = self.sanitize_base64_str(row['data'])
        count = self.increment_or_init_client_count(client)
        img_fn = self.image_fn_format.format(client, count)
        self.save_image(b64_image, client, img_fn)

    def write_error_logs(self, e: Exception, index: int, row: pd.Series):
        self.unsuccessful_conversions += 1
        with open(os.path.join(self.error_dir, self.error_fn_format.format(index)), 'w') as f:
            f.writelines('Exception: {}\n'.format(str(e)))
            f.writelines('Client: {}\n'.format(row['client']))
            f.writelines('Data: {}'.format(row['data']))

    def __call__(self, *args, **kwargs):
        df = pd.read_csv(self.csv_path)
        for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            try:
                self.write_image_file(row)
            except Exception as e:
                self.write_error_logs(e, index, row)

        if self.s3_dir_key is not None and self.s3_bucket is not None:
            self.write_batch_transform_input_file()

        self.print_summary(df.shape[0])

    def create_sub_directory(self, name: str):
        path = os.path.join(self.output_dir, name)
        if not os.path.isdir(path):
            os.mkdir(path)

        return path

    def print_summary(self, all_images: int):
        print('======== Script execution summary ========')
        print('The csv file contain {} image(s) belonging to {} client(s).\n{} image(s) failed to get converted to jpg file(s).'
              .format(all_images, len(self.counter.keys()), self.unsuccessful_conversions))
        if self.unsuccessful_conversions:
            print('please check the error logs to know the reason for failing to convert those images located in: {}'
                  .format(self.error_dir))

    def write_batch_transform_input_file(self):
        csv_file = open(os.path.join(self.output_dir, 'batch_transform_input.csv'), 'w')
        for client, image_count in self.counter.items():
            for index in range(1, image_count + 1):
                image_s3_key = os.path.join(self.s3_dir_key, client, self.image_fn_format.format(client, index))
                csv_file.write('{},{}\n'.format(self.s3_bucket, image_s3_key))
