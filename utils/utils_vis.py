from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


def save_image(img: np.ndarray, path_to_save: str) -> None:
    """Save the image to the specified path."""
    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
    Image.fromarray(img).save(path_to_save)

def save_fig(fig, path_to_save):
    fig.savefig(path_to_save, format='png', bbox_inches='tight')
    plt.close(fig)  # Close the figure to avoid displaying it in the notebook/output


def plot_artery_ann(vis, cnt_outer, cnts_mid, cnts_inner, cnts_hya = None, cnt_thick=2):
    # Create a copy of the visualization to avoid modifying the original
    vis_copy = vis.copy()
    
    # Draw the contours on the copy of the image
    cv2.drawContours(vis_copy, [cnt_outer], -1, [255, 0, 0], cnt_thick)  # Outer contour in red
    cv2.drawContours(vis_copy, cnts_mid, -1, [0, 255, 0], cnt_thick)    # Middle contours in green
    cv2.drawContours(vis_copy, cnts_inner, -1, [0, 0, 255], cnt_thick)  # Inner contours in blue
    if cnts_hya is not None:
        cv2.drawContours(vis_copy, cnts_hya, -1, [0, 255, 255], cnt_thick)  # Inner contours in blue

    # Return the modified copy
    return vis_copy
