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

import sys
sys.path.append(os.path.abspath('..'))
from utils.utils_data import Config
from utils.utils_vis import save_image
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_inference(model, dataloader, num_classes, save_dir, pred_suffix, device):
    total_batches = len(dataloader)

    for batch_id, (img_paths, inputs, _, w_oris, h_oris) in enumerate(dataloader):
        logging.info(f"Processing batch {batch_id+1}/{total_batches}")

        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = outputs.cpu().detach().numpy()

        for i, output in enumerate(outputs):
            img_path = img_paths[i]
            img_name = os.path.basename(img_path)
            output = np.argmax(output, axis = 0).astype(np.uint8)
            pred = (output*(255/(num_classes-1))).astype(np.uint8)
            pred_path = os.path.join(save_dir, img_name.replace("_ori.png", pred_suffix))
            save_image(pred, pred_path, (w_oris[i], h_oris[i]))

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
        task = config.get("task"),
        img_path_list = config.get("inference.inference_files"), 
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
    logging.info("Starting Inference process...")

    run_inference(model, dataloader, config.get('net.n_classes'), 
                  config.get('inference.save_dir'), config.get('inference.pred_suffix'), device)
    end_time = time.time()
    logging.info(f'Inference completed in {(end_time - start_time)/60:.2f} minutes.')

if __name__ == "__main__":
    main()
