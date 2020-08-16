import os
import random
from shutil import copyfile
from typing import Union

from .shared import get_init_extension_value, file_has_correct_ext

destination_sub_dirs = ['train', 'valid', 'test']


def split_file_list(file_list, sizes):
    length = len(file_list)
    start_index = 0
    split_files = []
    for size in sizes:
        start = int(length * start_index)
        start_index += size
        end = int(length * start_index)
        split_files.append(file_list[start:end])

    return split_files


def mv_keep_original(source, destination):
    copyfile(source, destination)


def mv_remove_original(source, destination):
    raise NotImplementedError()


def move_file(origin_dir, destination_dir, file, keep_original):
    # todo: implement keep_original = false
    if keep_original:
        src = os.path.join(origin_dir, file)
        dst = os.path.join(destination_dir, file)
        mv_keep_original(src, dst)


def move_files(file_lists, origin_dir, formatted_destination, keep_original):
    for index, file_list in enumerate(file_lists):
        for file in file_list:
            destination_path = formatted_destination.format(destination_sub_dirs[index])
            move_file(origin_dir, destination_path, file, keep_original)


def create_destination_dirs(destination_dir, number_splits):
    for i in range(number_splits):
        os.makedirs(os.path.join(destination_dir, destination_sub_dirs[i], 'images'))
        os.makedirs(os.path.join(destination_dir, destination_sub_dirs[i], 'labels'))


# TODO: make the function more generic, for now I am assuming that origin_dirs's length = 2 (image & label folders) and
#  destination folder will always have two sub_dirs (image & label)
#  also, destination_dirs depend entirely on size length: 1 -> train | 2 -> (train & valid) |
#  3 -> (train & valid & test) | else -> not supported.
# origin_dir = xxx/train
# destination_dir = yyy -> create yyy/{train, valid, test} -> under each sub-folder create images & labels folders
def shuffle_split_files(
        sizes: list,
        origin_dir: str,
        destination_dir: str,
        extensions: Union[str, list, None] = 'jpg',
        keep_original: bool = True):
    assert sum(sizes) == 1, 'sizes must add up to 1 (100%)'
    assert 1 <= len(sizes) <= 3, 'length of sizes must be between 1 and 3'
    create_destination_dirs(destination_dir, len(sizes))
    extensions = get_init_extension_value(extensions)
    file_list = [i for i in os.listdir(os.path.join(origin_dir, 'images')) if file_has_correct_ext(i, extensions)]
    random.shuffle(file_list)
    img_files, label_files = [[os.path.splitext(file)[0] + ext for file in file_list] for ext in ['.jpg', '.txt']]
    img_files_lists = split_file_list(img_files, sizes)
    label_files_lists = split_file_list(label_files, sizes)
    # print(img_files, img_files_lists)
    # print(label_files, label_files_lists)
    # return
    formatted_destination = os.path.join(destination_dir, '{}', 'images')
    move_files(img_files_lists, os.path.join(origin_dir, 'images'), formatted_destination, keep_original)
    formatted_destination = os.path.join(destination_dir, '{}', 'labels')
    move_files(label_files_lists, os.path.join(origin_dir, 'labels'), formatted_destination, keep_original)
