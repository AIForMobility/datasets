from typing import Union
from PIL import Image
import os


def file_has_correct_ext(filename: str, extension: Union[str, list, None]) -> bool:
    if extension is None:
        return True

    if isinstance(extension, str):
        check = lambda x: os.path.splitext(x)[1] == extension
    else:
        check = lambda x: os.path.splitext(x)[1] in extension

    return check(filename)


def get_init_extension_value(extension: Union[str, list, None]) -> Union[str, list]:
    def append_point_if_needed(ext: str) -> str:
        return '.' + ext if ext[0] != '.' else ext

    if isinstance(extension, str):
        return append_point_if_needed(extension)

    return [append_point_if_needed(ext) for ext in extension]


def get_image_dims(image) -> tuple:
    image = Image.open(image, 'r')  # type: Image.Image
    return image.width, image.height


def min_image_dims_in_dir(directory: str, extension: Union[str, list, None] = None) -> tuple:
    extension = get_init_extension_value(extension)
    width = -1
    height = -1
    images = [i for i in os.listdir(directory) if file_has_correct_ext(i, extension)]

    for image in images:
        w, h = get_image_dims(os.path.join(directory, image))
        width = w if w < width or width < 0 else width
        height = h if h < height or height < 0 else height

    return width, height

