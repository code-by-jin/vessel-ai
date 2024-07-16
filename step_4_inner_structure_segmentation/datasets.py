import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os


def rgb_to_mask_intra_arterial(rgb_image):
    # Define the mapping from RGB colors to class labels
    color_to_label = {
        (255, 0, 0): 1,   # Outer contour in red
        (0, 255, 0): 2,   # Middle contours in green
        (0, 0, 255): 3,   # Inner contours in blue
        (0, 0, 0): 0      # Background
    }
    label_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)
    for color, label in color_to_label.items():
        # Create a mask for each color matching
        matches = np.all(rgb_image == np.array(color, dtype=np.uint8), axis=-1)
        label_mask[matches] = label
    return label_mask


def rgb_to_mask_hyalinosis(rgb_image):
    # Define the mapping from RGB colors to class labels
    color_to_label = {
        (255, 0, 0): 0,   # Outer contour in red
        (0, 255, 0): 0,   # Middle contours in green
        (0, 0, 255): 0,   # Inner contours in blue
        (128, 0, 128): 1, # Hyalinosis contours in purple
        (0, 0, 0): 0      # Background
    }
    label_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)
    for color, label in color_to_label.items():
        # Create a mask for each color matching
        matches = np.all(rgb_image == np.array(color, dtype=np.uint8), axis=-1)
        label_mask[matches] = label
    return label_mask


class VesselDataset(Dataset):
    def __init__(self, task, img_path_list, test=False, mask_suffix=None, transform=None, target_transform=None, transform_seed=0):
        self.task = task
        self.img_path_list = img_path_list
        self.test = test
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.target_transform = target_transform
        self.transform_seed = transform_seed

    def __getitem__(self, idx):
        # Load images and masks
        img_path = self.img_path_list[idx]
        img = Image.open(img_path).convert("RGB")
        w_ori, h_ori = img.size
        img = np.array(img)
        if self.transform is not None:
            torch.manual_seed(self.transform_seed)
            img = self.transform(img)

        target = torch.zeros(1, dtype=torch.long)  # Placeholder tensor
        if not self.test:
            target_path = img_path.replace("_ori.png", self.mask_suffix)
            target = Image.open(target_path).convert("RGB")
            # Convert RGB mask to categorical labels
            target = np.array(target)
            if self.task == "hyalinosis":
                target = rgb_to_mask_hyalinosis(target)
            elif self.task == "intra_arterial":
                target = rgb_to_mask_intra_arterial(target)
            else:
                raise ValueError(f"Unknown task '{self.task}' provided.")

            if self.target_transform is not None:
                torch.manual_seed(self.transform_seed)
                target = self.target_transform(target)
        return img_path, img, target, w_ori, h_ori

    def __len__(self):
        return len(self.img_path_list)
    

# class VesselClassificationDataset(Dataset):
#     def __init__(self, img_path_list, test=False, transform=None, transform_seed=0):
#         self.img_path_list = img_path_list
#         self.test = test
#         self.transform = transform
#         self.transform_seed = transform_seed

#     def __getitem__(self, idx):
#         # Load images and masks
#         img_path = self.img_path_list[idx]
#         img = Image.open(img_path).convert("RGB")
#         w_ori, h_ori = img.size
#         img = np.array(img)
#         if self.transform is not None:
#             torch.manual_seed(self.transform_seed)
#             img = self.transform(img)

#         target = torch.zeros(1, dtype=torch.long)  # Placeholder tensor
#         if not self.test:
#             target_path = img_path.replace("_ori.png", self.mask_suffix)
#             target = Image.open(target_path).convert("RGB")
#             # Convert RGB mask to categorical labels
#             target = np.array(target)
#             target = rgb_to_mask(target)
#             if self.target_transform is not None:
#                 torch.manual_seed(self.transform_seed)
#                 target = self.target_transform(target)
#         return img_path, img, target, w_ori, h_ori

#     def __len__(self):
#         return len(self.img_path_list)
