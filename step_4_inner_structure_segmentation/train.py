import argparse
import logging
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from unet import UNet
from datasets import VesselDataset as Dataset
import time
from datetime import datetime

import sys
sys.path.append(os.path.abspath('..'))
from utils.utils_data import Config

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from datetime import datetime

def train_model(model, dataloaders, criterion, optimizer, scheduler,
                num_epochs, edge_weight, device, save_path):
    
    edge_weight=torch.tensor(edge_weight).to(device)
    best_loss = np.inf
    losses = {"train": [], "val": []}  # To store loss per epoch

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            for _, inputs, targets, _, _ in dataloaders[phase]:
                inputs, targets = inputs.to(device), targets.type('torch.LongTensor').to(device)
                optimizer.zero_grad() # Zero the parameter gradients
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets).mean()
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            losses[phase].append(epoch_loss)  # Append loss to the corresponding list

        logging.info(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {losses["train"][-1]:.4f}, Val Loss: {losses["val"][-1]:.4f}')
        
        # Save the best model based on validation loss
        if losses["val"][-1] < best_loss:
            best_loss = losses["val"][-1]
            logging.info(f'Saving new best model with Val Loss: {best_loss:.4f}')
            state = {'epoch': epoch + 1, 'model_dict': model.state_dict(), 'losses': losses}
            torch.save(state, save_path.replace(".pth", "_best.pth"))

        scheduler.step()
    
    # At the end of all epochs, save the final model and losses together
    logging.info('Saving the model and losses at the last epoch')
    final_state = {'epoch': epoch + 1,
                   'model_dict': model.state_dict(),
                   'losses': losses}
    torch.save(final_state, save_path.replace(".pth", f"_epoch_{epoch + 1}.pth"))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a U-Net model for image segmentation.")
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the configuration JSON file.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    return parser.parse_args()

def main():

    start_time = time.time()
    args = parse_arguments()

    # Load configuration
    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config = Config(config_path)
    device = torch.device(f'cuda:{config.get("training.gpu_id")}' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Initialize model
    model = UNet(
        n_classes=config.get('net.n_classes'),
        in_channels=config.get('net.in_channels'),
        padding=config.get('net.padding'),
        depth=config.get('net.depth'),
        wf=config.get('net.wf'),
        up_mode=config.get('net.up_mode'),
        batch_norm=config.get('net.batch_norm')
    ).to(device)
    logging.info(f"Model initialized")

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('training.lr'))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Decays LR by a factor of 0.1 every 10 epochs

    # Initialize criterion
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    
    transforms_dict = {
        "train": transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((256, 256)),  # Specify desired height and width
            transforms.ToTensor(),
        ]),
        "val": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),  # Specify desired height and width
            transforms.ToTensor(),
        ])
    }

    target_transform_dict = {
         "train": transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((256, 256)),  # Specify desired height and width
            lambda x: np.array(x, dtype=np.int32),  # Convert PIL Image to a NumPy array, maintaining integer labels
            transforms.ToTensor(),
            lambda x: torch.squeeze(x, 0).long()  # Remove channel dimension and convert to long dtype
        ]),
        "val": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),  # Specify desired height and width
            lambda x: np.array(x, dtype=np.int32),  # Convert PIL Image to a NumPy array, maintaining integer labels
            transforms.ToTensor(),
            lambda x: torch.squeeze(x, 0).long()  # Remove channel dimension and convert to long dtype
        ])
    } 
    # Initialize datasets
    datasets = {
        phase: Dataset(
            task = config.get("task"),
            img_path_list = config.get(f"data.{phase}_files"), 
            mask_suffix = "_mask.png",
            transform = transforms_dict[phase], 
            target_transform = target_transform_dict[phase]
        ) for phase in ["train", "val"]
    }

    # Initialize dataloaders
    dataloaders = {
        phase: DataLoader(
            dataset=datasets[phase],
            batch_size=config.get("training.batch_size"),
            shuffle=True,
            num_workers=4,
            pin_memory=True
        ) for phase in ["train", "val"]
    }
    logging.info(f"Datasets and dataloaders initialized")

    logging.info("Starting training process...")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join("models", f"{config.get('task')}_model_{timestamp}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_model(model, dataloaders, criterion, optimizer, scheduler,
                config.get('training.num_epochs'), config.get("training.edge_weight"), device, save_path)
    end_time = time.time()
    logging.info(f'Training completed in {(end_time - start_time)/60:.2f} minutes.')

if __name__ == "__main__":
    main()
