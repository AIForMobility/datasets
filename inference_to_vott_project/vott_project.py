import json
from typing import Dict, Tuple, Union

from inference_to_vott_project.functions import parse_pred_line, get_tags, create_full_asset


class VoTTProject:
    def __init__(self,
                 project_config_file: str,
                 source_con_path: str,
                 target_con_path: str,
                 predictions_csv: str,
                 labels_map: Dict[int, str],
                 s3_local_path_mapping: Tuple[str, str],
                 version: str = '2.2.0'):
        self.source_con_path = source_con_path
        self.target_con_path = target_con_path
        self.predictions_csv = predictions_csv
        self.project_config_file = project_config_file
        self.labels_map = labels_map
        self.version = version
        self.s3_local_path_mapping = s3_local_path_mapping
        self.project_config = {}
        self.assets = {}

    def update_project_config(self):
        with open(self.project_config_file, 'r') as f:
            self.project_config = json.load(f)

        self.project_config['tags'] = get_tags(list(self.labels_map.values()))
        self.project_config['exportFormat']['providerType'] = 'csv'
        self.project_config['activeLearningSettings'] = {
            'autoDetect': False,
            'predictTag': False,
            'modelPathType': 'coco'
        }

    def generate_assets(self):
        last_visited_asset = None
        self.project_config['assets'] = {}
        with open(self.predictions_csv, 'r') as file:
            for line in file.readlines():
                s3_path, bboxes = parse_pred_line(line)
                local_path = s3_path.replace(self.s3_local_path_mapping[0], self.s3_local_path_mapping[1])
                # full asset contains 3 things: asset, regions & version
                full_asset = create_full_asset(local_path, bboxes, self.labels_map, self.version)
                self.update_assets_and_config(full_asset)
                if last_visited_asset is None and full_asset is not None:
                    last_visited_asset = full_asset['asset']['id']

        if last_visited_asset is not None:
            self.project_config['lastVisitedAssetId'] = last_visited_asset

    def update_assets_and_config(self, full_asset: Union[dict, None]):
        if full_asset is None:
            return

        asset_id = full_asset['asset']['id']
        self.assets[asset_id] = full_asset
        self.project_config['assets'][asset_id] = full_asset['asset']
