from utils.b64_to_image_file import Base64ToImageFileConverter
import os


output_dir = '/Users/bothmena/Projects/datasets/after_rental_pictures_v2'
csv_path = os.path.join(output_dir, 'after_rental_pictures.csv')

converter = Base64ToImageFileConverter(csv_path,
                                       output_dir,
                                       s3_bucket='wb-inference-data',
                                       s3_dir_key='vehicle-detection/batch-transform-input/images/first-batch-transform/')
converter()
