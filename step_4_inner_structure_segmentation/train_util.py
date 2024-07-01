import os
import sys
import time

import random
random.seed(0)

import torch
torch.manual_seed(0)

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import numpy as np
np.random.seed(0)

from PIL import Image

from vessel_ai.step_4_inner_structure_segmentation.datasets import VesselDataset

sys.path.append(os.path.abspath('..'))
from utils.utils_vis import save_image


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred, numLabels):
    dice_list = []
    for index in range(numLabels):
        dice = dice_coef(y_true==index, y_pred==index)
        dice_list.append(dice)
    return dice_list # taking average


def train(device, root, net, epochs, batch_size, lr, reg, log_every_n=1):
    """
    Training a network
    :param net: Network for training
    :param epochs: Number of epochs in total.
    :param batch_size: Batch size for training.
    """
    print('==> Preparing data..')

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomAffine(30, translate=None, scale=(0.7,1.3), shear=20),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.Resize((256, 256)),  # Specify desired height and width
        transforms.ToTensor(),
        ])

    target_transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomAffine(30, translate=None, scale=(0.7,1.3), shear=20),
        transforms.Resize((256, 256)),  # Specify desired height and width
        lambda x: np.array(x, dtype=np.int32),  # Convert PIL Image to a NumPy array, maintaining integer labels
        transforms.ToTensor(),
        lambda x: torch.squeeze(x, 0).long()  # Remove channel dimension and convert to long dtype
    ])
    
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # Specify desired height and width
        transforms.ToTensor(),
        ])

    target_transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # Specify desired height and width
        lambda x: np.array(x, dtype=np.int32),  # Convert PIL Image to a NumPy array, maintaining integer labels
        transforms.ToTensor(),
        lambda x: torch.squeeze(x, 0).long()  # Remove channel dimension and convert to long dtype
    ])

    trainset = VesselDataset(root, train=True, transform = transform_train, target_transform= target_transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)

    valset = VesselDataset(root, train=False, transform = transform_test, target_transform= target_transform_test)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=16)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(net.parameters(),lr = lr)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[
        int(epochs*0.2), int(epochs*0.4), int(epochs*0.6), int(epochs*0.8)], gamma=0.5)

    global_steps = 0
    start = time.time()

    train_losses = np.zeros((epochs, ))
    val_losses = np.zeros((epochs, ))

    for epoch in range(epochs):
        """
        Start the training code.
        """
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        for batch_idx, (img_names, inputs, targets, w_ori, h_ori) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.type('torch.LongTensor').to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets).mean()
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            global_steps += 1

            if global_steps % log_every_n == 0:
                end = time.time()
                num_examples_per_second = log_every_n * batch_size / (end - start)
                print("[Step=%d]\tLoss=%.4f\t%.1f examples/second"
                      % (global_steps, train_loss / (batch_idx + 1), num_examples_per_second))
                start = time.time()
        train_losses[epoch] = train_loss / (batch_idx + 1)
        scheduler.step()

        """
        Start the val code.
        """
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (img_names, inputs, targets, w_ori, h_ori) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.type('torch.LongTensor').to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets).mean()
                val_loss += loss.item()
        print("Val Loss=%.4f"% (val_loss / (batch_idx + 1)))
        val_losses[epoch] = val_loss / (batch_idx + 1)
        print("Saving...")
        torch.save(net.state_dict(), "net.pt")
        np.save("train_losses.npy", train_losses)
        np.save("val_losses.npy", val_losses)


def resize_back(outputs, w_ori, h_ori):
    
    outputs = outputs.cpu().detach().numpy()
    outputs = np.argmax(outputs, axis = 1).astype(np.uint8)
    outputs = np.squeeze(outputs)
    im = Image.fromarray((outputs*(255/4)).astype(np.uint8))
    im = im.resize((w_ori, h_ori), resample=Image.NEAREST)
    return im

def test(device, root, net):
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # Specify desired height and width
        transforms.ToTensor(),
        ])

    target_transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # Specify desired height and width
        lambda x: np.array(x, dtype=np.int32),  # Convert PIL Image to a NumPy array, maintaining integer labels
        transforms.ToTensor(),
        lambda x: torch.squeeze(x, 0).long()  # Remove channel dimension and convert to long dtype
    ])
    
    testset = VesselDataset(root=root, train=False, transform = transform_test, target_transform= target_transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    net.eval()
    for batch_idx, (img_names, inputs, targets, w_ori, h_ori) in enumerate(testloader):
        assert(len(img_names) == 1)
        img_name = img_names[0]
        if batch_idx > 100: break
        inputs = inputs.to(device)
        outputs = net(inputs)

        pred_img = resize_back(outputs, w_ori, h_ori) 
        gt_img = np.squeeze(targets.cpu().detach().numpy())
        gt_img = Image.fromarray((gt_img*(255/4)).astype(np.uint8))
        gt_img = gt_img.resize((w_ori, h_ori), resample=Image.NEAREST) 
        pred_img.save(os.path.join("./output", img_name.replace(".png", '_inner_structure_pred.png')))
        gt_img.save(os.path.join("./output", img_name.replace(".png", '_inner_structure_gt.png')))