
# Standard library imports
import os
import sys
import argparse
import time
import logging

# Third-party imports
import pandas as pd
import numpy as np
import cv2

# Local module imports
sys.path.append(os.path.abspath('..'))
from utils.utils_post_process import post_process
from utils.utils_data import get_classifications, get_segmentations, get_measurements
from utils.utils_geometry import get_contours
from utils.utils_constants import (VESSEL_NEPTUNE_PAT_INFO_PATH as VESSEL_PAT_INFO_PATH, 
                                   CLASSIFICATION_PATH, SEGMENTATION_DIR,
                                   MEASUREMENTS_DIR, FEATURES_PATH)
from utils.utils_feature import extract_features

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_aspect_ratio(contour):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    return aspect_ratio


def calculate_convexity(contour):
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    contour_area = cv2.contourArea(contour)
    if contour_area == 0:
        return 0  # Avoid division by zero
    convexity = contour_area / hull_area
    return convexity


def extract_base_features(cnt_outer, cnts_middle, cnts_inner, cnts_hys):
    artery_area = cv2.contourArea(cnt_outer)
    lumen_area = sum(cv2.contourArea(contour) for contour in cnts_inner)
    intima_area = sum(cv2.contourArea(contour) for contour in cnts_middle) - lumen_area
    media_area = artery_area - lumen_area - intima_area
    hys_area = sum(cv2.contourArea(contour) for contour in cnts_hys)

    aspect_ratio = calculate_aspect_ratio(cnt_outer)
    convexity = calculate_convexity(cnt_outer)

    base_features = {
        'Artery Area': artery_area,
        'Log Artery Area': np.log(artery_area),
        'Media Area Ratio': media_area/artery_area,
        'Intima Area Ratio': intima_area/artery_area,
        'Lumen Area Ratio': lumen_area/artery_area,
        'Hyalinosis Area Ratio': hys_area/artery_area,
        'Aspect Ratio': aspect_ratio,
        'Convexity': convexity
    }
    return base_features

def extract_measurement_features(measurements_vessel, artery_area):
    # Initialize arrays for ratio calculations
    all_media = []
    all_intima = []
    all_ratio = []
    # Iterate over each row in the measurement DataFrame
    for m in measurements_vessel:
        m_media = np.array(m["Thickness_Media"])
        m_intima = np.array(m["Thickness_Intima"])

        # Example processing assuming m_media and m_intima are arrays
        m_wall = np.array([x + y if x >= 0 else x for x, y in zip(m_media, m_intima)])

        m_media, m_intima, m_ratio = post_process(m_media, m_intima, m_wall,
                                                  t_multi=15, t_open_lumen=30, t_mediam=15, t_average=15, 
                                                  artery_area=artery_area)
        all_media.extend(m_media)
        all_intima.extend(m_intima)
        all_ratio.extend(m_ratio)

    # Assuming all_media and all_intima are lists of arrays, we concatenate them to perform a global calculation
    features_intima, features_media, features_ratio = extract_features(all_media, all_intima, all_ratio)
    return {**features_intima, **features_media, **features_ratio}


def extract_features_slide(classifications, segmentations, measurements, slide_basename):
    slide_features = []
    for _, row in classifications.iterrows():
        img_name = row["Image Name"]
        bbox_x, bbox_y, bbox_width, bbox_height = map(int, row["Bounding Box"].split(","))  
        cnt_outer, cnts_middle, cnts_inner, cnts_hys = get_contours(segmentations, slide_basename, img_name,
                                                                    bbox_x, bbox_y, bbox_width, bbox_height)
        base_features = extract_base_features(cnt_outer, cnts_middle, cnts_inner, cnts_hys)

        measurements_vessel = measurements[img_name]
        if len(measurements_vessel) == 0:
            logging.warning(
                    f"No measurements for image {img_name} in slide {slide_basename}."
                    )
            measurement_features = {}
        else:
            measurement_features = extract_measurement_features(measurements_vessel, cv2.contourArea(cnt_outer))

        slide_features.append({**{"Slide Name": slide_basename}, **row.to_dict(), **base_features, **measurement_features})
    return slide_features


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process vessel images, compute measurements, and extract features.")
    parser.add_argument('--suffix', type=str, choices=[
        '_measurements', 
        '_measurements_exclude_hya_manual',
        '_measurements_exclude_hya_dl',
        '_measurements_lumen_convex',
        '_measurements_exclude_hya_manual_lumen_convex',
        '_measurements_exclude_hya_dl_lumen_convex'
    ], default='_measurements', help='Select the suffix for measurement files.')
    return parser.parse_args()


def main():
    start_time = time.time()
    args = parse_arguments()
    suffix = args.suffix
    pat_df = pd.read_csv(VESSEL_PAT_INFO_PATH)
    available_sheetnames = pd.ExcelFile(CLASSIFICATION_PATH).sheet_names
    logging.info(f"{len(pat_df)} slides selected, {len(pat_df) - len(available_sheetnames)} discarded, " 
                f"{len(available_sheetnames)} left for analysis.")
    
    excel_writer = pd.ExcelWriter(FEATURES_PATH.replace(".xlsx", f"{suffix}.xlsx"), engine='xlsxwriter')

    for i, slide_filename in enumerate(pat_df["WSI_Selected"]):
        logging.info(f"Processing: {i+1}/{len(pat_df)}: {slide_filename}")
        slide_basename = os.path.splitext(slide_filename)[0]

        classifications = get_classifications(CLASSIFICATION_PATH, slide_basename, available_sheetnames, remove_others=False)
        if classifications.empty:
            continue  # Skip to if no relevant data

        segmentations_path = os.path.join(SEGMENTATION_DIR, f"{slide_basename}.geojson")
        segmentations = get_segmentations(segmentations_path, clean=True)

        measurements_path = os.path.join(MEASUREMENTS_DIR, f"{slide_basename}{suffix}.json")
        measurements = get_measurements(measurements_path, clean=True)
        
        features = extract_features_slide(classifications, segmentations, measurements, slide_basename)
        features = pd.DataFrame(features)

        features.to_excel(excel_writer, sheet_name=slide_basename, index=False)
        worksheet = excel_writer.sheets[slide_basename]
        # Loop through the columns and set the column width as desired
        for i, col in enumerate(features.columns):
            # Set the column width
            column_len = max(features[col].astype(str).apply(len).max(), len(col)) + 2  # Adding a little extra space
            worksheet.set_column(i, i, column_len)

    excel_writer.close()

    end_time = time.time()
    logging.info(f"All slides processed and features extracted in {(end_time - start_time)/60:.2f} minutes.")

if __name__ == '__main__':
    main()