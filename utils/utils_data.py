import os
import sys
import pandas as pd
import geojson
import json

sys.path.append(os.path.abspath('..'))
from utils.utils_geometry import clean_geojson_annotations


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            self.config = json.load(file)
        logging.info(f"Configuration loaded from {config_file}")

    def get(self, path, default=None):
        keys = path.split('.')
        value = self.config
        for key in keys:
            if key in value:
                value = value[key]
            else:
                return default
        return value
    

def read_df_from_json(path_json):
    df = pd.read_json(path_json, orient="records", lines=True)
    return df


def get_classifications(classifications_path, sheet_name, available_sheets, remove_others=True):
    if sheet_name not in available_sheets:
        logging.info(f"Sheet {sheet_name} not found in the classifications file.")
        return pd.DataFrame()
    df = pd.read_excel(classifications_path, sheet_name=sheet_name)
    if remove_others:
        return df[df["Artery Type"] != "Others"]
    else:
        return df


def get_segmentations(segmentations_path, clean=True):
    with open(segmentations_path) as f:
        exported = geojson.load(f)
        annotations = exported['features']
    if clean:
        annotations = clean_geojson_annotations(annotations)
    return annotations