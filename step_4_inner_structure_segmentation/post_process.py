import pandas as pd
import numpy as np
import os
import sys
import cv2
from skimage import morphology, filters, img_as_float
sys.path.append(os.path.abspath('..'))

from utils.utils_constants import (COMBINED_CLASSIFICATION_PATH, 
                                   CROPPED_VESSELS_COMBINED_DIR)
from utils.utils_vis import save_image

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def rgb_to_mask_intima(rgb_image):
    # Define the mapping from RGB colors to class labels
    color_to_label = {
        (255, 0, 0): 0,   # Outer contour in red
        (0, 255, 0): 1,   # Middle contours in green
        (0, 0, 255): 1,   # Inner contours in blue
        (128, 0, 128): 1, # Hyalinosis contours in purple
        (0, 0, 0): 0      # Background
    }

    label_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)
    for color, label in color_to_label.items():
        # Create a mask for each color matching
        matches = np.all(rgb_image == np.array(color, dtype=np.uint8), axis=-1)
        label_mask[matches] = label
    return label_mask

def smooth_edges_gaussian(mask, sigma=1):
    mask_float = img_as_float(mask)  # Convert to float
    smoothed_mask = filters.gaussian(mask_float, sigma=sigma)
    return (smoothed_mask > 0.5).astype(int)  # Threshold back to binary

def post_process(mask, min_size_objects=400, min_size_holes=200):
    mask_bool = mask > 0
    cleaned_mask = morphology.remove_small_objects(mask_bool, min_size=min_size_objects)
    final_mask = morphology.remove_small_holes(cleaned_mask, area_threshold=min_size_holes)
    final_mask = np.uint8(final_mask) * 255  # Convert boolean to uint8
    return final_mask


combined_classifications = pd.read_csv(COMBINED_CLASSIFICATION_PATH)

for index, row in combined_classifications.iterrows():
    if index%50 == 0:
        logging.info(f"Processing {index + 1}/{len(combined_classifications)}: {row['Image Name']}")
    img_name = row["Image Name"]
    severity = row["Hyalinosis Severity"]  # Assuming the column is named 'Hyalinosis Severity'

    img_path = os.path.join(CROPPED_VESSELS_COMBINED_DIR, img_name.replace(".png", "_w_ann.png"))
    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    mask_path = os.path.join(CROPPED_VESSELS_COMBINED_DIR, img_name.replace(".png", "_mask.png"))
    mask = cv2.cvtColor(cv2.imread(mask_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = rgb_to_mask_intima(mask)

    pred_path = os.path.join(CROPPED_VESSELS_COMBINED_DIR, img_name.replace(".png", "_pred_hya.png"))
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    pred_filtered = post_process(pred)
    pred_filtered = smooth_edges_gaussian(pred_filtered, sigma=2)
    pred_filtered = np.where(mask, pred_filtered, 0)

    if np.sum(pred_filtered) == 0:
        save_image(pred_filtered.astype(np.uint8)*255, pred_path.replace(".png", "_processed.png"))
        continue

    save_image(pred_filtered.astype(np.uint8)*255, pred_path.replace(".png", "_processed.png"))
