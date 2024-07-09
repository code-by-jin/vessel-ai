import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew

def get_peak_prominence(data, peak_idx):

    left_idx = right_idx = peak_idx

    while (left_idx > 0 
           and ~np.isnan(data[left_idx - 1]) 
           and data[left_idx - 1] < data[left_idx]):
        left_idx -= 1

    while (right_idx < len(data) - 1 
           and ~np.isnan(data[right_idx + 1]) 
           and data[right_idx + 1] < data[right_idx]):
        right_idx += 1

    prominence = data[peak_idx] - min(data[left_idx], data[right_idx])
    return prominence, right_idx - left_idx

def circular_peaks(data, width_threshold=15):
    n = len(data)
    data_extended = np.concatenate([data, data, data])
    data_extended_no_nan = np.where(np.isnan(data_extended), 
                                    np.nanmin(data) - 1, data_extended)
    
    peak_indices_extended, _ = find_peaks(data_extended_no_nan)
    
    peak_indices = []
    prominences = []
    widths = []
    for idx in peak_indices_extended:
        prominence, width = get_peak_prominence(data_extended, idx)
        if n <= idx < 2 * n and (width_threshold is None or width >= width_threshold):
            peak_indices.append(idx % n)
            prominences.append(prominence)
            widths.append(width)
    
    if len(peak_indices) == 0:
        return None, None

    peak_heights = data[np.array(peak_indices)]

    peak_properties = {
        'peak_heights': peak_heights,
        'prominences': np.array(prominences),
    }
    return np.array(peak_indices), peak_properties


def get_features(l_thick, prefix): 
    dict_features = {}
    l_thick_valid = [x for x in l_thick if x>=0]

    arr_thick = np.array(l_thick_valid)
    if prefix == "Ratio":
        arr_thick = arr_thick[arr_thick >= 0.05]
        arr_thick = arr_thick[arr_thick <= 20]
    dict_features[prefix+" Average"] = np.mean(arr_thick)
    dict_features[prefix+" Median"] = np.median(arr_thick)
    dict_features[prefix+" Variance"] = np.var(arr_thick)

    peak_indices, peak_properties = circular_peaks(np.array([x if x >=0 else np.nan for x in l_thick]))
    if peak_indices is not None:
        dict_features["Vis "+prefix+" Peak Indice"] = peak_indices[np.argmax(peak_properties["peak_heights"])]
        dict_features[prefix+" Peak Height"] = np.max(peak_properties["peak_heights"])
        dict_features[prefix+" Peak Prominence"] = np.max(peak_properties["prominences"])
    else:
        dict_features["Vis "+prefix+" Peak Indice"] = None
        dict_features[prefix+" Peak Height"] = 0
        dict_features[prefix+" Peak Prominence"] = 0
    return dict_features


def extract_features(thick_media, thick_intima, thick_ratio):
    
    features_intima = get_features(thick_intima, "Intima")
    features_media = get_features(thick_media, "Media")
    features_ratio = get_features(thick_ratio, "Ratio")
    
    return features_intima, features_media, features_ratio
    
