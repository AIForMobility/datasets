from utils.b64_to_image_file import Base64ToImageFileConverter
import os


output_dir = '/Users/bothmena/Projects/datasets/after_rental_pictures_v2'
csv_path = os.path.join(output_dir, 'after_rental_pictures_sample.csv')

converter = Base64ToImageFileConverter(csv_path, output_dir)
converter()
