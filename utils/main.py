from utils import min_image_dims_in_dir, shuffle_split_files

main_run = 'dims'  # 'split'

###################################
#    min_image_dims_in_dir        #
###################################

if main_run == 'dims':
    # directory = '/Users/bothmena/Projects/datasets/vehicles_object_detection/yolo_v5/train/images/'
    directory = '/Users/bothmena/Projects/AIForMobility/datasets/yolo_labels/yolo_v3_to_v5/sample/input_images'
    width, height = min_image_dims_in_dir(directory, ['jpg'])

    print('min width = {} / min height = {}'.format(width, height))

###################################
#    min_image_dims_in_dir        #
###################################

if main_run == 'split':
    original_dir = '/Users/bothmena/Projects/datasets/vehicles_object_detection/yolo_v5/train'
    destination = '/Users/bothmena/Projects/datasets/vehicles_object_detection/yolo_v5_train_valid'

    shuffle_split_files([.8, .2], original_dir, destination)
