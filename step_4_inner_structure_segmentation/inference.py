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
from PIL import Image

import sys
sys.path.append(os.path.abspath('..'))
from utils.utils_data import Config
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def run_inference(model, dataloader, device):
    for img_paths, inputs, _, w_ori, h_ori in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = outputs.cpu().detach().numpy()

        for i, output in enumerate(outputs):
            img_path = img_paths[i]
            output = np.argmax(output, axis = 0).astype(np.uint8)
            pred_img = Image.fromarray((output*(255/4)).astype(np.uint8))
            pred_img = pred_img.resize((w_ori, h_ori), resample=Image.NEAREST)
            pred_img.save(img_path.replace("_ori.png", '_pred.png'))

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
    device = torch.device(f'cuda:{config.get("inference.gpu_id")}' if torch.cuda.is_available() else 'cpu')
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

    model_path = config.get('inference.model_path')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage) 
    model.load_state_dict(checkpoint["model_dict"])
    model.eval()

    logging.info(f"Model initialized")

    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # Specify desired height and width
        transforms.ToTensor(),
        ])
  
    # Initialize datasets
    dataset = Dataset(
        img_path_list = config.get(f"inference.inference_files"), 
        test = True,
        transform = transform)


    # Initialize dataloaders
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.get("inference.batch_size"),
        shuffle=False,
        num_workers=4,
        pin_memory=True
        ) 
    
    logging.info(f"Datasets and dataloaders initialized")

    logging.info("Starting training process...")
    # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # save_path = os.path.join("models", f"{config.get('data.dataname')}_model_{timestamp}.pth")
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)

    run_inference(model, dataloader, device)
    end_time = time.time()
    logging.info(f'Training completed in {(end_time - start_time)/60:.2f} minutes.')

if __name__ == "__main__":
    main()
