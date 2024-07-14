
# Standard library imports
import os
import sys
import logging

# Third-party imports
import pandas as pd
import numpy as np
import cv2

# Local module imports
sys.path.append(os.path.abspath('..'))
from utils.utils_post_process import post_process
from utils.utils_vis import save_image, plot_artery_ann
from utils.utils_data import get_classifications, get_segmentations
from utils.utils_geometry import get_contours, is_contour_intersecting_or_within
from utils.utils_measure import measure_thickness
from utils.utils_constants import (VESSEL_NEPTUNE_PAT_INFO_PATH as VESSEL_PAT_INFO_PATH, 
                                   CLASSIFICATION_PATH, SEGMENTATION_DIR,
                                   MEASUREMENTS_PATH, CROPPED_VESSELS_DIR, CROPPED_VESSELS_COMBINED_DIR)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_vessel_measurements(outer_contour, middle_contours, inner_contours, hys_contours,
                              grayscale_image, angle_width=15, vis=None, 
                              classification_info={}):
    """Calculate geometric features of vessels based on given contours."""
    measurements = []
    artery_area = cv2.contourArea(outer_contour)
    lumen_area = sum(cv2.contourArea(contour) for contour in inner_contours)
    intima_area = sum(cv2.contourArea(contour) for contour in middle_contours) - lumen_area
    media_area = artery_area - lumen_area - intima_area
    hys_area = sum(cv2.contourArea(contour) for contour in hys_contours)

    common_info = {
        **classification_info,
        'Area_Media': media_area,
        'Area_Intima': intima_area,
        'Area_Lumen': lumen_area,
        'Area_Hys': hys_area
    }

    for idx_inner, inner_contour in enumerate(inner_contours):
        for idx_middle, middle_contour in enumerate(middle_contours):
            if is_contour_intersecting_or_within(inner_contour, middle_contour):
                exclude = (middle_contours[:idx_middle] + middle_contours[idx_middle+1:] +
                           inner_contours[:idx_inner] + inner_contours[idx_inner+1:] + hys_contours)
                curr_area_lumen = cv2.contourArea(inner_contour)
                curr_area_intima = cv2.contourArea(middle_contour) - curr_area_lumen
                thick_media, thick_intima = measure_thickness(
                    outer_contour, middle_contour, inner_contour, grayscale_image,
                    angle_width=20, exclude=exclude, vis=None
                )

                thick_wall_raw = np.array([x + y if x >= 0 else x for x, y in zip(thick_media, thick_intima)])
                vis_helper, _, _ = post_process(thick_media, thick_intima, thick_wall_raw,
                                                t_multi=15, t_open_lumen=30, t_mediam=15, t_average=15, artery_area=artery_area)
                _, _ = measure_thickness(
                    outer_contour, middle_contour, inner_contour, grayscale_image,
                    angle_width=20, exclude=exclude, vis=vis, vis_helper = vis_helper
                )
    
                measurement = {
                    **common_info, 
                    'Thickness_Media': thick_media, 
                    'Thickness_Intima': thick_intima, 
                    'Curr_Area_Intima': curr_area_intima, 
                    'Curr_Area_Lumen': curr_area_lumen
                }
                measurements.append(measurement)

    return measurements


def get_hya_pred_contour(img_name):
    
    pred_path = os.path.join(CROPPED_VESSELS_COMBINED_DIR, img_name.replace(".png", "_pred_hya_processed.png"))
    pred_filtered = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(pred_filtered.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt.squeeze() for cnt in contours if len(cnt) > 4]


def process_slide(classifications, segmentations, slide_basename):
    slide_measurements = []
    for _, row in classifications.iterrows():
        image_path = os.path.join(CROPPED_VESSELS_DIR, row["Artery Type"], row["Image Name"].replace(".png", "_ori.png"))
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) 
        # img_w_ann = cv2.cvtColor(cv2.imread(image_path.replace("_ori.png", "_w_ann.png")), cv2.COLOR_BGR2RGB) 
        bbox_x, bbox_y, bbox_width, bbox_height = map(int, row["Bounding Box"].split(","))  
        cnt_outer, cnts_middle, cnts_inner, cnts_hys = get_contours(segmentations, slide_basename, row["Image Name"],
                                                                    bbox_x, bbox_y, bbox_width, bbox_height)
        
        cnts_hys = get_hya_pred_contour(row["Image Name"])
        cnts_inner = [cv2.convexHull(cnt).squeeze() for cnt in cnts_inner]

        img_w_ann = plot_artery_ann(img, cnt_outer, cnts_middle, cnts_inner, cnts_hya = cnts_hys, cnt_thick=2)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        measurements = calculate_vessel_measurements(
            cnt_outer, cnts_middle, cnts_inner, cnts_hys, img_gray,
            angle_width=20, vis=img_w_ann, classification_info=row.to_dict()
        )

        save_image(img_w_ann, image_path.replace(".png", "_w_measurements_in_convex_hya_pred.png"))
        slide_measurements.extend(measurements)
    return slide_measurements


pat_df = pd.read_csv(VESSEL_PAT_INFO_PATH)
pat_df = pat_df[pat_df["WSI_Selected"].notna() 
                & pat_df["ESRDorEGFR40BX_LR"].notna() 
                & pat_df["DaysBXtoESRDorEGFR40_LR"].notna()]

available_sheetnames = pd.ExcelFile(CLASSIFICATION_PATH).sheet_names
logging.info(f"{len(pat_df)} slides selected, {len(pat_df) - len(available_sheetnames)} discarded, " 
             f"{len(available_sheetnames)} left for analysis.")

collected_measurements = []
for i, slide_filename in enumerate(pat_df["WSI_Selected"]):
    logging.info(f"Processing: {i+1}/{len(pat_df)}: {slide_filename}")
    slide_basename = os.path.splitext(slide_filename)[0]
    classifications = get_classifications(CLASSIFICATION_PATH, slide_basename, available_sheetnames, remove_others=False)
    if classifications.empty:
        continue  # Skip to if no relevant data
    segmentations_path = os.path.join(SEGMENTATION_DIR, f"{slide_basename}.geojson")
    segmentations = get_segmentations(segmentations_path, clean=True)

    slide_measurements = process_slide(classifications, segmentations, slide_basename)
    collected_measurements.extend(slide_measurements)

    df = pd.DataFrame(collected_measurements)
    df.to_json(MEASUREMENTS_PATH.replace(".json", "_convex_hya_pred.json"), orient='records', lines=True)