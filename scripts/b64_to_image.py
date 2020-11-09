"""
a script that converts a dataset of images saved in csv file to image files
the csv file is expected to have two columns:
- client: which is a string: name of the client owning the image
- data: an image encoded in base64

the script will read the csv file and save all images as a jpg file in the following structure:
`output_dir`
  |- `client`
    |- `{client}_{:06d}.jpg (example: easyway_000123.jpg)
"""
import pandas as pd
import os
import tqdm
from base64 import b64decode


dataset_dir = '/Users/bothmena/Projects/datasets/after_rental_pictures_v2/'
csv_path = os.path.join(dataset_dir, 'after_rental_pictures.csv')
output_dir = os.path.join(dataset_dir, 'images')
error_dir = os.path.join(dataset_dir, 'errors')
df = pd.read_csv(csv_path)
counter = {}


def get_client_count(client: str):
    try:
        counter[client] += 1
        return counter[client]
    except KeyError:
        counter[client] = 1
        return 1


def get_abs_file_path(client: str, filename: str):
    dir_path = os.path.join(output_dir, client)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return os.path.join(dir_path, filename)


def save_image(data: str, client: str, filename: str):
    img_data = b64decode(data)
    file_path = get_abs_file_path(client, filename)
    with open(file_path, 'wb') as f:
        f.write(img_data)


def write_image_file(row: pd.Series):
    client = row['client']
    b64_image = row['data'][2:-1]
    count = get_client_count(client)
    img_fn = '{}_{:06d}.jpg'.format(client, count)
    save_image(b64_image, client, img_fn)


def write_error_logs(e: Exception, index: int, row: pd.Series):
    with open(os.path.join(error_dir, 'error_{:07d}.txt'.format(index))) as f:
        f.write(str(e))
        f.write('client: {}'.format(row['client']))
        f.write(row['data'])


def main():
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        try:
            write_image_file(row)
        except Exception as e:
            write_error_logs(e, index, row)


if __name__ == '__main__':
    main()
