
# Standard library imports
import os
import sys
import argparse
import time
import json

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
                                   MEASUREMENTS_DIR, CROPPED_VESSELS_DIR, CROPPED_VESSELS_COMBINED_DIR)

# Logging configuration
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_vessel_measurements(outer_contour, middle_contours, inner_contours, hys_contours, 
                                  grayscale_image, angle_width=20, vis=None):
    
    measurements = []
    artery_area = cv2.contourArea(outer_contour)
    for idx_inner, inner_contour in enumerate(inner_contours):
        for idx_middle, middle_contour in enumerate(middle_contours):
            if not is_contour_intersecting_or_within(inner_contour, middle_contour):
                continue

            exclude = (middle_contours[:idx_middle] + middle_contours[idx_middle+1:]
                       + inner_contours[:idx_inner] + inner_contours[idx_inner+1:]
                       + hys_contours)
            
            thick_media, thick_intima = measure_thickness(
                outer_contour, middle_contour, inner_contour, grayscale_image,
                angle_width=angle_width, exclude=exclude, vis=None
            )

            curr_area_lumen = cv2.contourArea(inner_contour)
            curr_area_intima = cv2.contourArea(middle_contour) - curr_area_lumen

            thick_wall_raw = np.array([x + y if x >= 0 else x for x, y in zip(thick_media, thick_intima)])
            vis_helper, _, _ = post_process(thick_media, thick_intima, thick_wall_raw, 
                                            t_multi=15, t_open_lumen=30, t_mediam=15, t_average=15, 
                                            artery_area=artery_area)
            
            _, _ = measure_thickness(
                outer_contour, middle_contour, inner_contour, grayscale_image,
                angle_width=angle_width, exclude=exclude, vis=vis, vis_helper = vis_helper
            )

            measurements.append({
                'Thickness_Media': thick_media, 
                'Thickness_Intima': thick_intima, 
                'Curr_Area_Intima': curr_area_intima, 
                'Curr_Area_Lumen': curr_area_lumen
                })

    return measurements


def get_hya_pred_contour(img_name):
    
    pred_path = os.path.join(CROPPED_VESSELS_COMBINED_DIR, img_name.replace(".png", "_pred_hya_processed.png"))
    pred_filtered = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(pred_filtered.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt.squeeze() for cnt in contours if len(cnt) > 4]


def calculate_slide_measurements(classifications, segmentations, slide_basename, 
                                 exclude_hyalinosis=None, lumen_transfer=None, suffix="_ann"):
    slide_measurements = {}
    for _, row in classifications.iterrows():
        img_name = row["Image Name"]
        image_path = os.path.join(CROPPED_VESSELS_DIR, row["Artery Type"], img_name.replace(".png", "_ori.png"))
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        # img_w_ann = cv2.cvtColor(cv2.imread(image_path.replace("_ori.png", "_w_ann.png")), cv2.COLOR_BGR2RGB) 
        bbox_x, bbox_y, bbox_width, bbox_height = map(int, row["Bounding Box"].split(","))  
        cnt_outer, cnts_middle, cnts_inner, cnts_hys = get_contours(segmentations, slide_basename, img_name,
                                                                    bbox_x, bbox_y, bbox_width, bbox_height)
        
        if exclude_hyalinosis == "dl":
            cnts_hys = get_hya_pred_contour(img_name)
        elif exclude_hyalinosis != "Manual":
            cnts_hys = []

        if lumen_transfer == "convex":
            cnts_inner = [cv2.convexHull(cnt).squeeze() for cnt in cnts_inner]

        img_w_ann = plot_artery_ann(img, cnt_outer, cnts_middle, cnts_inner, cnts_hya = cnts_hys, cnt_thick=2)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if not cnts_middle or not cnts_inner:
            logging.warning(
                    f"Missing Intima or Lumen contours for {slide_basename} in image {img_name}."
                )
            measurements = []
        else:
            measurements = calculate_vessel_measurements(
                cnt_outer, cnts_middle, cnts_inner, cnts_hys, img_gray,
                angle_width=20, vis=img_w_ann
            )

        slide_measurements[img_name] = measurements
        save_image(img_w_ann, image_path.replace(".png", suffix + ".png"))

    return slide_measurements


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process vessel images and compute measurements.")
    parser.add_argument('--lumen_transfer', type=str, choices=[None, 'convex'], default=None, help='Method to use for transforming the contour of the lumen.')
    parser.add_argument('--exclude_hyalinosis', type=str, choices=[None, 'manual', 'dl'], default=None, help='Method to handle hyalinosis measurements intersection.')
    return parser.parse_args()


def create_output_suffix(base, exclude_hyalinosis, lumen_adjustment):
    """Generate a file suffix that describes the processing options used."""
    suffix = base
    if exclude_hyalinosis is not None:
        suffix += f"_exclude_hya_{exclude_hyalinosis}"
    if lumen_adjustment is not None:
        suffix += f"_lumen_{lumen_adjustment}"
    return suffix


def main():
    start_time = time.time()
    args = parse_arguments()
    suffix = create_output_suffix("_measurements", args.exclude_hyalinosis, args.lumen_transfer)
    
    pat_df = pd.read_csv(VESSEL_PAT_INFO_PATH)

    available_sheetnames = pd.ExcelFile(CLASSIFICATION_PATH).sheet_names
    logging.info(f"{len(pat_df)} slides selected, {len(pat_df) - len(available_sheetnames)} discarded, " 
                f"{len(available_sheetnames)} left for analysis.")

    for i, slide_filename in enumerate(pat_df["WSI_Selected"]):
        logging.info(f"Processing: {i+1}/{len(pat_df)}: {slide_filename}")
        slide_basename = os.path.splitext(slide_filename)[0]
        classifications = get_classifications(CLASSIFICATION_PATH, slide_basename, available_sheetnames, 
                                              remove_others=False)
        if classifications.empty:
            continue  # Skip to if no relevant data
        segmentations_path = os.path.join(SEGMENTATION_DIR, f"{slide_basename}.geojson")
        segmentations = get_segmentations(segmentations_path, clean=True)

        measurements = calculate_slide_measurements(classifications, segmentations, slide_basename, 
                                                    args.exclude_hyalinosis, args.lumen_transfer, suffix)
        
        measurements_path = os.path.join(MEASUREMENTS_DIR, f"{slide_basename}{suffix}.json")
        with open(measurements_path, 'w') as file:
            json.dump(measurements, file)

    end_time = time.time()
    logging.info(f'Measurements completed in {(end_time - start_time)/60:.2f} minutes with '
                 f'lumen_transfer={args.lumen_transfer or "None"}, '
                 f'exclude_hyalinosis={args.exclude_hyalinosis or "None"}')

if __name__ == '__main__':
    main()