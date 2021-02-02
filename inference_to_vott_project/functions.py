import json
import os
import random
import string
from typing import Dict

from utils.images import get_image_dims

LETTERS_AND_DIGITS = string.ascii_letters + string.digits
# Full list of colors: https://en.wikipedia.org/wiki/List_of_colors:_A%E2%80%93F
COLORS = [
    '0048BA',
    'B0BF1A',
    '7CB9E8',
    'C0E8D5',
    'B284BE',
    '72A0C1',
    'EDEAE0',
    'F0F8FF',
    'C46210',
    'EFDECD',
    'E52B50',
    '9F2B68',
    'F19CBB',
]


def generate_uuid(length: int):
    return ''.join((random.choice(LETTERS_AND_DIGITS) for _ in range(length)))


def create_connection(name: str):
    return {
        'name': name,
        'providerType': 'localFileSystemProxy',
        'providerOptions': {
            'providerType': generate_uuid(312)
        },
        'id': generate_uuid(9),
    }


def get_tags(labels: list):
    tags = []
    colors = COLORS.copy()
    random.shuffle(COLORS)
    for i, label in enumerate(labels):
        tags.append({
            'name': label,
            'color': '#{}'.format(colors[i])
        })

    return tags


def parse_pred_line(line: str) -> (str, list):
    path, *rest = line.split(',')
    bboxes_str = ','.join(rest)
    return path, json.loads(bboxes_str)


def create_asset(path: str, width: int, height: int):
    return {
        'format': os.path.splitext(path)[-1][1:],
        'id': generate_uuid(32),
        'name': os.path.basename(path) + '.jpg',  # todo remove .jpg
        'path': 'file:{}.jpg'.format(path),  # todo remove .jpg
        'size': {
            'width': width,
            'height': height
        },
        # state values: 1 -> visited but not tagged | 2 -> visited & tagged | probably 0 -> not visited & not tagged.
        'state': 2,
        'type': 1
    }


def create_asset_regions(bboxes: list, label_maps: Dict[int, str]) -> list:
    regions = []
    for (x_min, y_min, x_max, y_max, _, label) in bboxes:
        regions.append({
            'id': generate_uuid(9),
            'type': 'RECTANGLE',
            'tags': [label_maps[int(label)]],
            'boundingBox': {
                'height': y_max - y_min,
                'width': x_max - x_min,
                'left': x_min,
                'top': y_min
            },
            'points': [
                {'x': x_min, 'y': y_min},
                {'x': x_max, 'y': y_min},
                {'x': x_max, 'y': y_max},
                {'x': x_min, 'y': y_max}
            ]
        })

    return regions


def create_full_asset(path: str, bboxes: list, labels_map: Dict[int, str], version: str):
    if not len(bboxes):
        return None

    width, height = get_image_dims(path)

    return {
        'asset': create_asset(path, width, height),
        'regions': create_asset_regions(bboxes, labels_map),
        'version': version
    }
