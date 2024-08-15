#!/bin/bash

# Define arrays for lumen and hyalinosis options
lumen_options=("--lumen_transfer convex")
# hyalinosis_options=("" "--exclude_hyalinosis manual" "--exclude_hyalinosis dl")
hyalinosis_options=("--exclude_hyalinosis manual")

# Loop through all combinations of options
for lumen in "${lumen_options[@]}"; do
    for hyalinosis in "${hyalinosis_options[@]}"; do
        echo "Running: python measure_thickness.py $lumen $hyalinosis"
        python measure_thickness.py $lumen $hyalinosis
    done
done
