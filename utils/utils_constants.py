
ARTERY_TYPES = ["Arterioles", "Interlobular Arteries", "Arcuate Arteries"]
DISEASE_TYPES = ["Arteriosclerosis", "Hyalinosis"]
# Mapping dictionary for severity
CLASSIFICATION_SEVERITY_MAPPING = {
    '0 - No': 0,
    '1 - Mild': 1,
    '2 - Moderate': 2,
    '3 - Severe': 3
}

# Constants for file paths to clinical info
NEPTUNE_PAT_INFO_PATH = "/DataMount/NEPTUNE/Clinical_Data/Barisoni_NEPTUNE_clinical_20230620.csv"
NEPTUNE_WSI_INFO_PATH= "/DataMount/NEPTUNE/Clinical_Data/NEPTUNE Spreadsheet 20230216.xlsx"
VESSEL_NEPTUNE_PAT_INFO_PATH = "/DataMount/NEPTUNE/Clinical_Data/Barisoni_NEPTUNE_clinical_20230620_Vessel_20240625.csv"
VESSEL_NEPTUNE_PAT_INFO_W_SCORE_PATH = "/DataMount/NEPTUNE/Clinical_Data/Barisoni_NEPTUNE_clinical_20230620_Vessel_20240615_W_SCORE.csv"
VESSEL_NEPTUNE_PAT_INFO_W_SCORE_W_FEATURE_PATH = "/DataMount/NEPTUNE/Clinical_Data/Barisoni_NEPTUNE_clinical_20230620_Vessel_20240615_W_SCORE_W_FEATURE.csv"

# Classification Path
CLASSIFICATION_PATH = "/workspace/vessel_ai/artery_classification/Neptune_Artery_Classification_Sheets.xlsx"
SEGMENTATION_DIR = "/DataMount/NEPTUNE/Vessel_Project/data_selection/ann_geojson/all"
VESSEL_SEGMENTATION_REF_PATH = "/DataMount/NEPTUNE/Vessel_Project/data_selection/Normal Vessels GT Generation .xlsx"
INNER_SEGMENTATION_REF_DIR = "/DataMount/NEPTUNE/Vessel_Project/data_selection/ann_geojson/batch_0_manual_annotated"

MEASUREMENTS_PATH = "/DataMount/NEPTUNE/Vessel_Project/thickness_measurements/thickness_0603.json"
FEATURES_PATH = "/DataMount/NEPTUNE/Vessel_Project/features_0603.xlsx"

# Data paths
TRI_CASE_DIR = "/DataMount/NEPTUNE/Vessel_Project/TRI/"
CROPPED_VESSELS_DIR = "/DataMount/NEPTUNE/Vessel_Project/Cropped_Vessels"


