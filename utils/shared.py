from typing import Union
import os


def get_init_extension_value(extension: Union[str, list, None]) -> Union[str, list]:
    def append_point_if_needed(ext: str) -> str:
        return '.' + ext if ext[0] != '.' else ext

    if isinstance(extension, str):
        return append_point_if_needed(extension)

    return [append_point_if_needed(ext) for ext in extension]


def file_has_correct_ext(filename: str, extension: Union[str, list, None]) -> bool:
    if extension is None:
        return True

    if isinstance(extension, str):
        check = lambda x: os.path.splitext(x)[1] == extension
    else:
        check = lambda x: os.path.splitext(x)[1] in extension

    return check(filename)
