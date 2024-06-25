import os
import sys
import pandas as pd

sys.path.append(os.path.abspath('..'))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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