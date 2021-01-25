import os


def create_versioned_dataset_dir(formatted_path: str) -> str:
    v = 1
    final_dir_path = formatted_path.format(v)
    while os.path.isdir(final_dir_path):
        v += 1
        final_dir_path = formatted_path.format(v)

    print('final_dir_path', final_dir_path)
    os.mkdir(final_dir_path)

    return final_dir_path
