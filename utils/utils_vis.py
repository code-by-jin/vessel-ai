from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


def save_image(img: np.ndarray, path_to_save: str, resize = None) -> None:
    """Save the image to the specified path."""
    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
    img =  Image.fromarray(img)
    if resize is not None:
        img = img.resize(resize, resample=Image.NEAREST)
    img.save(path_to_save)

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
        cv2.drawContours(vis_copy, cnts_hya, -1, [128, 0, 128], cnt_thick)  # Inner contours in blue

    # Return the modified copy
    return vis_copy


def gallery_view(images, titles, cols=5):
    # Number of images to show per page/view
    num_images = len(images)
    rows_per_view = 1  # Show one row at a time

    # Calculate the number of views needed
    total_views = (num_images + cols - 1) // cols

    for view in range(total_views):
        start_index = view * cols
        end_index = min(start_index + cols, num_images)
        # fig, axs = plt.subplots(rows_per_view, cols, figsize=(15, 5 * rows_per_view))
        fig, axs = plt.subplots(end_index-start_index, rows_per_view, figsize=(5, 5*(end_index-start_index)))

        axs = axs.ravel()

        for i in range(cols):
            index = start_index + i
            if index < end_index:
                image = images[index]
                # Rotate image if width is greater than height
                # if image.shape[0] > image.shape[1]:  # image.shape gives (height, width, channels)
                if image.shape[1] > image.shape[0]:  # image.shape gives (height, width, channels)
                    image = np.rot90(image)  # Rotate 90 degrees
                axs[i].imshow(image)
                axs[i].set_title(titles[index], fontsize=20)
                # axs[i].set_title(titles[index], fontsize=15)

                axs[i].axis('off')
            else:
                continue
                # axs[i].axis('off')
        plt.tight_layout()
        # Save the figure to file
        plt.show()


def plot_two_in_one_col(thick_media, thick_intima, y_label=True, p_idx_intima=None, p_idx_media=None, path_to_save=None):
    
    plt.figure(figsize=(10, 5)) 
    plt.plot([x if x>=0 else None for x in thick_media], label="Media")
    plt.plot([x if x>=0 else None for x in thick_intima], label="Intima")
        
    if p_idx_media is not None:
        plt.scatter(p_idx_media, np.array(thick_media)[p_idx_media], marker="x", s=100)
    if p_idx_intima is not None:
        plt.scatter(p_idx_intima, np.array(thick_intima)[p_idx_intima], marker="x", s=100)
    discard_samples = 0
    start = None
    for i, x in enumerate(thick_intima):
        if x < 0 and start is None:
            start = i
        elif x >= 0 and start is not None:
            discard_samples += 1
            if discard_samples == 1:
                plt.axvspan(start, i-1, alpha=0.4, facecolor='gray', label="Discard")
            else:
                plt.axvspan(start, i-1, alpha=0.4, facecolor='gray')
            start = None
    
    # If the last chunk of -2 goes until the end of the list
    if start is not None:
        plt.axvspan(start, i, alpha=0.4, facecolor='gray')
     
    plt.xlabel("Angle", fontsize=20)
    plt.xticks(np.arange(0, 361, step=120), fontsize=15)
    
    plt.yticks(ticks=np.arange(0, 0.5, step=0.2), fontsize=15)
    plt.ylabel("Thickness", fontsize=20)
    plt.legend(fontsize=20, framealpha=0.5, loc='upper right')
    plt.tight_layout()
    if path_to_save:
        plt.savefig(path_to_save)
    # plt.show()