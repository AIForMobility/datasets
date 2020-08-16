from typing import Union
from PIL import Image
import os
from .shared import get_init_extension_value, file_has_correct_ext


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
