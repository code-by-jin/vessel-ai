# Basic imports
import os
import sys
import pandas as pd 

sys.path.append(os.path.abspath('..'))
from utils.utils_data import get_veesel_sheets
from utils.utils_constants import (ARTERY_TYPES,
                                   DISEASE_TYPES,
                                   VESSEL_NEPTUNE_PAT_INFO_PATH as VESSEL_PAT_INFO_PATH, 
                                   VESSEL_NEPTUNE_PAT_INFO_W_SCORE_PATH as VESSEL_PAT_INFO_W_SCORE_PATH,
                                   CLASSIFICATION_PATH
                                   )

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_series_stat(series):
    if series.empty:
        return {
            "Max": None, 
            "Mean": None, 
            "Median": None, 
            "75th": None,
            "NonZeroPct": None
        }
    return {
        "Max": series.max(), 
        "Mean": series.mean(), 
        "Median": series.median(), 
        "75th": series.quantile(0.75),
        "NonZeroPct": (series > 0).sum() / len(series)
    }


def update_stat(pat_df, index, classifications, artery_type):
    pat_df.loc[index, f'Num_{artery_type}'.replace(" ", "_")] = len(classifications)
    for disease_type in DISEASE_TYPES:
        col = f"{disease_type} Severity"
        severity_series = classifications[col]

        stats = get_series_stat(severity_series)
        for stat_name, stat_value in stats.items():
            pat_df.loc[index, f'{stat_name}_{col}_in_{artery_type}'.replace(" ", "_")] = stat_value


pat_df = pd.read_csv(VESSEL_PAT_INFO_PATH)
available_sheetnames = pd.ExcelFile(CLASSIFICATION_PATH).sheet_names

for i, (index, row) in enumerate(pat_df.iterrows()):
    slide_filename = row["WSI_Selected"]
    logging.info(f"Processing: {i+1}/{len(pat_df)}: {slide_filename}")

    slide_basename = os.path.splitext(slide_filename)[0]
    classifications = get_veesel_sheets(CLASSIFICATION_PATH, slide_basename, available_sheetnames, remove_others=True)
    if classifications.empty:
        continue  # Skip to if no relevant data

    update_stat(pat_df, index, classifications, "All_Arteries")
    
    for artery_type in ARTERY_TYPES:
        update_stat(pat_df, index, 
                    classifications[classifications['Artery Type'] == artery_type], 
                    artery_type)

pat_df.to_csv(VESSEL_PAT_INFO_W_SCORE_PATH, index=False)