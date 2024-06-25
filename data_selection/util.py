import os
import pandas as pd
import openslide

def identify_wsi_based_on_prefix(wsi_df, prefixes, stain):
    """
    Identifies Whole Slide Images (WSIs) based on a list of prefixes for file names.
    
    Parameters:
    - wsi_df: DataFrame containing information about the WSIs.
    - prefixes: List of prefixes for file names to match against 'old_filename' and 'new_filename' in the WSI data.
    - stain: Specific stain type to filter the WSIs by.
    
    Returns:
    - A set of unique biopsy IDs associated with the prefixes.
    - A dictionary mapping each unique biopsy ID to its corresponding WSI file name that matches the given stain.
    """
    seg_biopsies = set()
    biopsy_to_wsi_map = {}
    
    for prefix in prefixes:
        # Filter the DataFrame based on whether 'old_filename' or 'new_filename' starts with the prefix
        
        mask = wsi_df['old_filename'].str.startswith(prefix) | wsi_df['new_filename'].str.startswith(prefix)
        filtered_df = wsi_df[mask]

        if filtered_df.empty:
            print(f"Warning: Prefix '{prefix}' is not found in WSI Info.")
            continue

        unique_biopsy_ids = filtered_df['biopsyid'].dropna().unique()
        
        if len(unique_biopsy_ids) != 1:
            status = "no unique" if unique_biopsy_ids.size == 0 else "multiple"
            print(f"Notice: Prefix '{prefix}' is associated with {status} biopsy ID(s).")
            continue
            
        biopsy_id = unique_biopsy_ids[0]
        seg_biopsies.add(biopsy_id)
    
        # Select WSI File 
        stain_filtered_df = filtered_df[filtered_df['stain'] == stain]
        
        if len(stain_filtered_df) == 1:  # Exactly one file with the required stain
            selected_file = stain_filtered_df.iloc[0]['old_filename'] or stain_filtered_df.iloc[0]['new_filename']
            biopsy_to_wsi_map[biopsy_id] = selected_file
        else:
            print(f"Notice: Prefix '{prefix}' is associated with {len(stain_filtered_df)} {stain} Stained WSI File(s).")
    return seg_biopsies, biopsy_to_wsi_map


def update_flag_and_check_missing(pat_df, biopsy_set, column_name):
    """
    Update clinical DataFrame with a flag for given biopsy IDs and check for any missing in clinical DataFrame.

    Parameters:
    - pat_df: The Patient DataFrame to be updated.
    - biopsy_set: Set of biopsy IDs to update the flag for.
    - column_name: The name of the column to update in clinical DataFrame.
    """
    # Set flag based on biopsy IDs
    pat_df[column_name] = pat_df['BiopsyID'].isin(biopsy_set)
    
    # Find and report missing biopsy IDs
    missing_biopsies = biopsy_set - set(pat_df['BiopsyID'])
    if missing_biopsies:
        print(f"Missing biopsy IDs for {column_name}:", missing_biopsies)
        # Add missing biopsy IDs with other columns as NaN
        for biopsy_id in missing_biopsies:
            # Create a new row with specified biopsy ID, set the flag to True for the column, and NaN for others
            new_row = {'BiopsyID': biopsy_id, column_name: True, "Not_in_PAT_Info": 1}
            # Append the new row to the DataFrame
            pat_df = pd.concat([pat_df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        print(f"All biopsy IDs for {column_name} are present in pat_info.")

    return pat_df


def check_if_file_exists(case_dir, filename):
    """Check if a file exists in the given directory."""
    path = os.path.join(case_dir, filename)
    return os.path.exists(path)


def check_if_file_openable(case_dir, filename):
    """Check if an OpenSlide file is openable."""
    try:
        path = os.path.join(case_dir, filename)
        slide = openslide.OpenSlide(path)
        slide.read_region((0, 0), 0, (10, 10))
        return True
    except (openslide.OpenSlideError, FileNotFoundError):
        return False
    
    
def get_wsi_files_by_biopsy_id_and_stain(wsi_df, stain, biopsy_id):
    filtered_df = wsi_df[(wsi_df["biopsyid"].astype(str) == str(biopsy_id)) & (wsi_df["stain"] == stain)]
    if not filtered_df.empty:
        filenames = filtered_df['old_filename'].combine_first(filtered_df['new_filename']).dropna().values
        return list(filenames)
    else:
        return []