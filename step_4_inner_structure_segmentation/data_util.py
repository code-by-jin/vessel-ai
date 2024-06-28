import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os


def rgb_to_mask(rgb_image):
    # Define the mapping from RGB colors to class labels
    color_to_label = {
        (255, 0, 0): 1,   # Outer contour in red
        (0, 255, 0): 2,   # Middle contours in green
        (0, 0, 255): 3,   # Inner contours in blue
        (128, 0, 128): 4, # Hyalinosis contours in purple
        (0, 0, 0): 0      # Background
    }
    label_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)
    for color, label in color_to_label.items():
        # Create a mask for each color matching
        matches = np.all(rgb_image == np.array(color, dtype=np.uint8), axis=-1)
        label_mask[matches] = label
    return label_mask


class VesselDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.imgs = sorted([s for s in os.listdir(root) if s.endswith("_ori.png")])
        self.targets = sorted([s for s in os.listdir(root) if s.endswith("_mask.png")])

    def __getitem__(self, idx):
        # Load images and masks
        img_path = os.path.join(self.root, self.imgs[idx])
        target_path = os.path.join(self.root, self.targets[idx])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        target = Image.open(target_path).convert("RGB")
        w_ori, h_ori = target.size

        # Convert RGB mask to categorical labels
        target = np.array(target)
        target = rgb_to_mask(target)
        
        if self.transform is not None:
            torch.manual_seed(0)
            img = self.transform(img)
        if self.target_transform is not None:
            torch.manual_seed(0)
            target = self.target_transform(target)
        return self.imgs[idx], img, target, w_ori, h_ori

    def __len__(self):
        return len(self.imgs)
    

# # Define the path to your dataset
# root_path = os.path.join(CROPPED_VESSELS_DIR, "Arterioles")

# # Create an instance of the dataset
# dataset = VesselDataset(root=root_path, train=True, transform = img_transform, target_transform= mask_transform)

# # Function to convert a tensor to a numpy array for visualization
# def tensor_to_numpy(tensor):
#     return tensor.permute(1, 2, 0).numpy()

# # Set up matplotlib figures
# fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
# fig.subplots_adjust(hspace=0.3, wspace=0.1)

# for i in range(3):
#     # Get data from dataset
#     _, img, mask, _, _ = dataset[i]  # Get the i-th sample

#     # Convert tensors to numpy arrays for visualization
#     img = tensor_to_numpy(img)
#     mask = tensor_to_numpy(mask)

#     # Plotting
#     axs[i, 0].imshow(img)
#     axs[i, 0].set_title(f'Original Image {i}')
#     axs[i, 0].axis('off')

#     axs[i, 1].imshow(mask, cmap='gray')
#     axs[i, 1].set_title(f'Mask Image {i}')
#     axs[i, 1].axis('off')

# plt.show()
