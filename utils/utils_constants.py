
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
CLASSIFICATION_PATH = "/DataMount/NEPTUNE/Vessel_Project/data_selection/Neptune_Artery_Classification_Sheets.xlsx"
COMBINED_CLASSIFICATION_PATH = "/DataMount/NEPTUNE/Vessel_Project/data_selection/Neptune_Artery_Classification_Combined.csv"
ANALYSIS_DOC_PATH = "/DataMount/NEPTUNE/Vessel_Project/data_selection/Artery_Analysis_Report.docx"


SEGMENTATION_DIR = "/DataMount/NEPTUNE/Vessel_Project/data_selection/ann_geojson/all"
VESSEL_SEGMENTATION_REF_PATH = "/DataMount/NEPTUNE/Vessel_Project/data_selection/Normal Vessels GT Generation .xlsx"
INNER_SEGMENTATION_REF_DIR = "/DataMount/NEPTUNE/Vessel_Project/data_selection/ann_geojson/batch_0_manual_annotated"

MEASUREMENTS_PATH = "/DataMount/NEPTUNE/Vessel_Project/thickness_measurements/thickness_0707.json"
FEATURES_PATH = "/DataMount/NEPTUNE/Vessel_Project/features_0707.xlsx"

# Data paths
TRI_CASE_DIR = "/DataMount/NEPTUNE/Vessel_Project/TRI/"
CROPPED_VESSELS_DIR = "/DataMount/NEPTUNE/Vessel_Project/Cropped_Vessels"
CROPPED_VESSELS_COMBINED_DIR = "/DataMount/NEPTUNE/Vessel_Project/Cropped_Vessels_Combined"

# Labelbox Keys, need to be remove when making the repo public
LABELBOX_API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja2J1NDB3a2ExMzl2MDgzMDV0bjZ1ejB2Iiwib3JnYW5pemF0aW9uSWQiOiJja2J1NDB3anY1M3ZiMDcxM2psNms1aGowIiwiYXBpS2V5SWQiOiJjbHVpbHU0dzQwMDUwMDd4cmc0bnUxdGUzIiwic2VjcmV0IjoiZDM4OWZmOThhMjFkMWZhYTY1ZWRiYWYyOWZmMTNjYjUiLCJpYXQiOjE3MTIwNzU5ODUsImV4cCI6MjM0MzIyNzk4NX0.bpJc5JcxnbP58I3KvAPehFDL9MAsDV4yCUUCqixfoTw'
PROJECT_KEY_1 = 'cltkig4hn00vi07wshjvt0s6e'
PROJECT_KEY_2 = 'clubq24ks0ld407x6700kb8yy'
PROJECT_KEY_REMOVE = 'clv9tfnzj0pvz08vmff6v979p'


