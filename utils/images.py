from typing import Union, Tuple, List
from PIL import Image
import os
from .shared import get_init_extension_value, file_has_correct_ext


def get_image_dims(image_path: str) -> tuple:
    image = Image.open(image_path, 'r')  # type: Image.Image
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


def normalize_bbox_coordinates(image_path: str, bbox) -> Tuple[float, ...]:
    x_min, y_min, x_max, y_max = bbox
    width, height = get_image_dims(image_path)

    normalized_bbox = x_min / width, y_min / height, x_max / width, y_max / height
    return tuple([round(f, 5) for f in list(normalized_bbox)])


def xyxy_to_cxcywh(bbox) -> Tuple[float, ...]:
    """
    converts bbox from format (x_min, y_min, x_max, y_max) to format (center_x, center_y, width, height)
    """
    xmn, ymn, xmx, ymx = bbox
    w, h = xmx - xmn, ymx - ymn
    cx, cy = xmn + w/2, ymn + h/2

    return cx, cy, w, h
