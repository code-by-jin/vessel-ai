from typing import List, Dict, Tuple
import numpy as np
import cv2
import logging


def adjust_coordinates(cnt, offset):
    """
    Adjust contours coordinates by a given offset.
    :param cnt: numpy array of contour points.
    :param offset: tuple (x_offset, y_offset) to adjust the contour coordinates.
    :return: Adjusted contour coordinates.
    """
    return cnt - np.array([[offset]])


def get_outer_boundary_of_polygon(polygon_coords: List[List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    """Extracts and returns the outer boundary coordinates of a Polygon."""
    return polygon_coords[0]


def select_largest_polygon_from_multipolygon(multipolygon_coords: List[List[List[Tuple[float, float]]]]) -> List[Tuple[float, float]]:
    """Selects and returns the largest polygon from a MultiPolygon."""
    # Transform each multipolygon into its largest polygon by vertex count
    multipolygon_coords = [get_outer_boundary_of_polygon(poly) for poly in multipolygon_coords]
    largest_polygon = max(multipolygon_coords, key=lambda poly: len(poly))
    return largest_polygon


def clean_geojson_annotations(annotations: List[Dict]) -> List[Dict]:
    """Simplifies geometry for GeoJSON annotations."""
    for ann in annotations:
        geom_type = ann['geometry']['type']
        coords = ann['geometry']['coordinates']
        if ann["properties"]["classification"]["name"].startswith("H"):
            ann["properties"]["classification"]["name"] = "Hyalinosis"
        
        if geom_type == "Polygon":
            ann['geometry']['coordinates'] = get_outer_boundary_of_polygon(coords)
        elif geom_type == "MultiPolygon":
            ann['geometry']['coordinates'] = select_largest_polygon_from_multipolygon(coords)
        else:
            logging.warning(f"Unsupported geometry type: {geom_type}")
    return annotations


# def clean_geojson_annotations_for_qpdata(annotations: List[Dict]) -> List[Dict]:
#     """Simplifies geometry for GeoJSON annotations."""
#     for ann in annotations:
#         geom_type = ann['geometry']['type']
#         coords = ann['geometry']['coordinates']
        
#         if geom_type == "Polygon":
#             ann['geometry']['coordinates'] = [get_outer_boundary_of_polygon(coords)]
#         elif geom_type == "MultiPolygon":
#             ann['geometry']['type'] = "Polygon"
#             ann['geometry']['coordinates'] = [select_largest_polygon_from_multipolygon(coords)]
#         else:
#             logging.warning(f"Unsupported geometry type: {geom_type}")
#     return annotations


# def calculate_contour_bounds(cnt: np.ndarray, img_h: int, img_w: int, border_size: float = 0.0) -> Tuple[int, int, int, int]:
#     """
#     Calculate the bounding rectangle for a contour with an added border, adjusting for image dimensions.
#     The border is calculated as a ratio of the smaller dimension of the bounding rectangle if less than 1,
#     otherwise as an absolute pixel value.

#     :param cnt: Contour array.
#     :param img_h: Height of the image.
#     :param img_w: Width of the image.
#     :param border_size: If less than 1, treated as a ratio of the bounding rectangle's smaller dimension;
#                         if 1 or greater, treated as an absolute border size in pixels.
#     :return: Tuple of (xmin, ymin, width, height) representing the adjusted bounding rectangle.
#     """
#     x, y, w, h = cv2.boundingRect(cnt)
#     # Determine if border_size is a ratio or a direct pixel value
#     border = border_size if border_size >= 1 else int(border_size * min(w, h))

#     xmin, ymin = max(x - border, 0), max(y - border, 0)
#     xmax, ymax = min(x + w + border, img_w), min(y + h + border, img_h)
#     adjusted_w, adjusted_h = xmax - xmin, ymax - ymin
#     return xmin, ymin, adjusted_w, adjusted_h


# def calculate_combined_bounds(contours: List[np.ndarray], img_h: int, img_w: int, border_size: float = 0.0) -> Tuple[int, int, int, int]:
#     """
#     Calculate the minimal bounding rectangle for a list of contours, incorporating a border around each,
#     and adjust for image dimensions. Returns the position and size of the combined bounding rectangle.
    
#     The border is applied based on the border_size parameter, which can be either a ratio of the
#     smaller dimension of the bounding rectangle if less than 1, or an absolute pixel value if 1 or greater.

#     :param contours: List of contour arrays.
#     :param img_h: Height of the image.
#     :param img_w: Width of the image.
#     :param border_size: If less than 1, treated as a ratio of the bounding rectangle's smaller dimension;
#                         if 1 or greater, treated as an absolute border size in pixels.
#     :return: Tuple of (xmin, ymin, width, height) representing the adjusted bounding rectangle.
#     """
#     if not contours:
#         return 0, 0, img_w, img_h  # Return full image dimensions if no contours are present.

#     xmin, ymin, xmax, ymax = img_w, img_h, 0, 0
#     for cnt in contours:
#         x, y, w, h = calculate_contour_bounds(cnt, img_h, img_w, border_size)
#         xmin, ymin = min(xmin, x), min(ymin, y)
#         xmax, ymax = max(xmax, x + w), max(ymax, y + h)

#     adjusted_w, adjusted_h = xmax - xmin, ymax - ymin
#     return xmin, ymin, adjusted_w, adjusted_h


# def get_cnts_inside(ann, cnt_outer, target):
#     cnts_inner_list = []
#     for i, ann_i in enumerate(ann):
#         ann_type = get_ann_type(ann_i)
#         if ann_type == target:
#             # check if inside or intersec
#             cnt_inner = ann_i["geometry"]["coordinates"]
#             if cnt_polygon_test(cnt_inner, cnt_outer):
#                 cnts_inner_list.append(cnt_inner)
#     return cnts_inner_list

# def cnt_polygon_test(cnt1, cnt2):
#     # check if cnt1 inside/cross cnt2    
#     for point in cnt1:        
#         if cv2.pointPolygonTest(cnt2, (int(point[0]), int(point[1])), False) >= 0: return True
#     return False


# def cnt_polygon_test_any(cnt_1, cnts_2) -> bool:
#     for cnt_2 in cnts_2:
#         if cnt_polygon_test(cnt_1, cnt_2):
#             return True
#     return False


# def get_ann_type(ann_i):
#     if "classification" in ann_i["properties"] and "name" in ann_i["properties"]["classification"]:
#         return ann_i["properties"]["classification"]["name"]
#     else:
#         print("STH WRONG")
        # return None # noise
    

# def get_outer_cnt(annotations, bounding_box):
#     x, y, w, h = bounding_box
#     cnts_outer = []
#     for ann in annotations:
#         if ann["properties"]["classification"]["name"] != "Media": continue
#         cnt = np.array(ann['geometry']['coordinates'], dtype=np.int32)
#         x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
#         if x_cnt < x or y_cnt < y or (x_cnt + w_cnt) > (x + w) or (y_cnt + h_cnt) > (y + h): continue
#         if (w > 1.2 * w_cnt) or (h > 1.2 * h_cnt): continue
#         cnts_outer.append(cnt)
#     return cnts_outer



def get_contours_by_classification(segmentations, filter_fn, classification="Media"):
    filtered_contours = []
    for annotation in segmentations:
        coordinates = np.array(annotation['geometry']['coordinates'], dtype=np.int32)

        if annotation["properties"]["classification"]["name"] == classification and filter_fn(coordinates):
            filtered_contours.append(coordinates)
    return filtered_contours


def is_contour_match_bounds(contour, bounding_box, size_ratio=1.2):
    """Check if the contour matches bounding box constraints and size ratio."""
    x, y, w, h = bounding_box
    x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(contour)
    within_bounds = (x_cnt >= x) and (y_cnt >= y) and ((x_cnt + w_cnt) <= (x + w)) and ((y_cnt + h_cnt) <= (y + h))
    fits_size_ratio = (w <= size_ratio * w_cnt) and (h <= size_ratio * h_cnt)
    return within_bounds and fits_size_ratio


def is_contour_intersecting_or_within(cnt_iner, cnt_outer):
    """Determine if an inner contour intersects or is completely within an outer contour."""
    return any(cv2.pointPolygonTest(cnt_outer, (int(point[0]), int(point[1])), False) >= 0 for point in cnt_iner)


def offset_contours(contours, offset):
    """Applies an offset to contour coordinates."""
    return [(contour - np.array([offset])).astype(np.int32).squeeze() for contour in contours]


def get_contours(segmentations, slide_basename, image_name, bbox_x, bbox_y, bbox_width, bbox_height):
    # Retrieve outer contours matching specific bounds conditions
    outer_contours = get_contours_by_classification(
        segmentations,
        lambda contour: is_contour_match_bounds(
            contour, (bbox_x, bbox_y, bbox_width, bbox_height)
        ),
        "Media"
    )

    if len(outer_contours) != 1:
        raise Exception(f"Expected one outer contour for {slide_basename}, found {len(outer_contours)}.")

    outer_contour = outer_contours[0]
    middle_contours = get_contours_by_classification(
        segmentations,
        lambda contour: is_contour_intersecting_or_within(contour, outer_contour),
        "Intima"
    )

    # Retrieve inner contours that intersect or are within the outer contour
    inner_contours = get_contours_by_classification(
        segmentations,
        lambda contour: is_contour_intersecting_or_within(contour, outer_contour),
        "Lumen"
    )

    hyal_contours = get_contours_by_classification(
        segmentations,
        lambda contour: is_contour_intersecting_or_within(contour, outer_contour),
        "Hyalinosis"
    )

    # Offset the contours to adjust for the bounding box extraction
    adjusted_outer = offset_contours([outer_contour], (bbox_x, bbox_y))[0]
    adjusted_middle = offset_contours(middle_contours, (bbox_x, bbox_y))
    adjusted_inner = offset_contours(inner_contours, (bbox_x, bbox_y))
    adjusted_hya = offset_contours(hyal_contours, (bbox_x, bbox_y))

    return adjusted_outer, adjusted_middle, adjusted_inner, adjusted_hya
