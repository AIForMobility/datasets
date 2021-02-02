import os
import json
from inference_to_vott_project.vott_project import VoTTProject

project_name = 'testing_script_attempt_1'
base_dir = '/Users/bothmena/Projects/AIForMobility/datasets/inference_to_vott_project'
source_con_path = os.path.join(base_dir, 'vott_source')
target_con_path = os.path.join(base_dir, 'vott_target')
predictions_csv = os.path.join(source_con_path, 'queue_part.csv.out')
labels_map = {
    0: 'E_SCOOTER',
    1: 'CAR',
    2: 'MOTORCYCLE',
    3: 'BICYCLE',
    4: 'SEGWAY',
    5: 'LICENCE_PLATE',
}

print('Step 1: Project details:')
if project_name is None:
    print('Project name: ')
    project_name = input()

if source_con_path is None:
    print('Source connection path (directory containing images): ')
    source_con_path = input()

if target_con_path is None:
    print('Target connection path (directory containing images): ')
    target_con_path = input()

if predictions_csv is None:
    print('Path to CSV file containing the predictions: ')
    predictions_csv = input()

if labels_map is None:
    raise Exception('please define the labels map')

project_config_path = os.path.join(target_con_path, '{}.vott'.format(project_name))
# while os.path.isfile(project_config_path):
#     print('Project already exists, enter new name: ')
#     project_name = input()
#     project_config_path = os.path.join(target_con_path, '{}.vott'.format(project_name))

s3_local_path_mapping = 's3://wb-inference-data/vehicle-detection/ping-test-images', source_con_path
project = VoTTProject(project_config_file=project_config_path,
                      source_con_path=source_con_path,
                      target_con_path=target_con_path,
                      predictions_csv=predictions_csv,
                      labels_map=labels_map,
                      s3_local_path_mapping=s3_local_path_mapping)

print('Step 2: Creating a VoTT project')
print('\tPlease open VoTT and create a new project with the given info:')
print('\t2.1 - Display name: "{}"'.format(project_name))
print('\t2.2 - Source Connection [Select an existing one having the following folder path or add a new one with the following data]:')
print('\t\tDisplay name: "{}_source_connection"'.format(project_name))
print('\t\tProvider: "Local File Storage"'.format(project_name))
print('\t\tFolder Path: "{}"'.format(source_con_path))
print('\t2.3 - Select the source connection you just created: "{}_source_connection"'.format(project_name))
print('\t2.4 - Target Connection [Select an existing one having the following folder path or add a new one with the following data]:')
print('\t\tDisplay name: "{}_target_connection"'.format(project_name))
print('\t\tProvider: "Local File Storage"'.format(project_name))
print('\t\tFolder Path: "{}"'.format(source_con_path))
print('\t2.5 - Select the target connection you just created: "{}_target_connection"'.format(project_name))
print('\t2.6 - Save Project')
print('when you are done press enter in the console:')
_ = input()

print('Step 3: Changing Export Settings')
print('\tClick on the "Export Icon" in the left side-nav and make the following changes:')
print('\t3.1 - Provider: Comma Separated Values (CSV)')
print('\t3.2 - Asset State: Only Tagged Assets')
print('\t3.3 - Uncheck "Include Images" Options')
print('\t3.4 - Save Export Settings, Save the project and quit VoTT.')
print('when you are done press enter in the console:')
_ = input()
print(' ')
print(' ')

print('Step 5: Updating project configuration')
print('\t5.1 - Updating project configurations')
project.update_project_config()
print('\t5.2 - Generating assets')
project.generate_assets()

print('\t5.3: Remove the project from Recent Projects (right side-nav)')
print('when you are done press enter in the console:')
_ = input()
print('\t5.4: QUIT VoTT')
print('when you are done press enter in the console:')
_ = input()

print('\t5.4 - Overwriting project vott file')
with open(project_config_path, 'w') as f:
    f.write(json.dumps(project.project_config))

print('\t5.5 - Writing asset json files')
for asset_id, asset in project.assets.items():
    with open(os.path.join(target_con_path, '{}-asset.json'.format(asset_id)), 'w') as f:
        f.write(json.dumps(asset))

print('Step 6: Open VoTT, and visualize the output of your model and improve predictions')
