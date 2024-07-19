#!/bin/bash

# Define the path to the Python script, adjust as necessary if not in the same directory
PYTHON_SCRIPT_PATH="./extract_features.py"

# Define all possible suffix options
SUFFIX_OPTIONS=(
    "_measurements"
    "_measurements_exclude_hya_manual"
    "_measurements_exclude_hya_dl"
    "_measurements_lumen_convex"
    "_measurements_exclude_hya_manual_lumen_convex"
    "_measurements_exclude_hya_dl_lumen_convex"
)

# Loop through all suffix options and run the Python script for each one
for suffix in "${SUFFIX_OPTIONS[@]}"
do
    echo "Running the script with suffix option: $suffix"
    python extract_features_vessel.py --suffix "$suffix"
done

echo "All processes completed."
