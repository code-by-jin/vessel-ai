#!/usr/bin/python
# -*- coding: utf-8 -*-
# import necessary packages

import os
import sys

import numpy as np
import pandas as pd
import openslide

sys.path.append(os.path.abspath('..'))
from utils.utils_data import get_classifications, get_segmentations
from utils.utils_geometry import get_contours

from utils.utils_vis import plot_artery_ann, save_image
from utils.utils_constants import (CLASSIFICATION_SEVERITY_MAPPING, 
                                   VESSEL_NEPTUNE_PAT_INFO_PATH as VESSEL_PAT_INFO_PATH, 
                                   CLASSIFICATION_PATH, SEGMENTATION_DIR, 
                                   TRI_CASE_DIR, CROPPED_VESSELS_DIR)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

pat_df = pd.read_csv(VESSEL_PAT_INFO_PATH)

available_sheetnames = pd.ExcelFile(CLASSIFICATION_PATH).sheet_names

for i, (_, row) in enumerate(pat_df.iterrows()):
    slide_filename = row["WSI_Selected"]
    slide_basename = os.path.splitext(slide_filename)[0]

    logging.info(f"Processing: {i+1}/{len(pat_df)} {slide_filename}")
    slide_path = os.path.join(TRI_CASE_DIR, slide_filename)
    slide = openslide.OpenSlide(slide_path)

    classifications = get_classifications(CLASSIFICATION_PATH, slide_basename, available_sheetnames, remove_others=False)
    if classifications.empty:
        continue  # Skip to if no relevant data
    classifications['Arteriosclerosis Severity'] = classifications['Arteriosclerosis Severity'].map(CLASSIFICATION_SEVERITY_MAPPING)

    segmentations_path = os.path.join(SEGMENTATION_DIR, f"{slide_basename}.geojson")
    segmentations = get_segmentations(segmentations_path, clean=True)

    for i, (_, row)in enumerate(classifications.iterrows()):
        bbox_x, bbox_y, bbox_width, bbox_height = map(int, row["Bounding Box"].split(","))  
        cnt_outer, cnts_middle, cnts_inner, cnts_hys = get_contours(segmentations, slide_basename, row["Image Name"], 
                                                                    bbox_x, bbox_y, bbox_width, bbox_height)
        img = np.array(slide.read_region((bbox_x, bbox_y), 0, (bbox_width, bbox_height)).convert("RGB"))
        path_img_to_save = os.path.join(CROPPED_VESSELS_DIR, row["Artery Type"], row["Image Name"])
        save_image(img, path_img_to_save)
        img_w_ann = plot_artery_ann(img, cnt_outer, cnts_middle, cnts_inner, cnts_hys)
        save_image(img_w_ann, path_img_to_save.replace(".png", "_w_ann.png"))
        mask = plot_artery_ann(np.zeros(img.shape, dtype=np.uint8), cnt_outer, cnts_middle, cnts_inner, cnts_hys, -1)
        save_image(mask, path_img_to_save.replace(".png", "_mask.png"))
