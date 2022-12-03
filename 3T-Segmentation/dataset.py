import os
import numpy as np
import nibabel as nib 
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

class WMH_Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #print(f"self images {self.images}")
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        #print(f"mask path {mask_path}")
        #print(mask_path)

        '''image = np.array(Image.open(img_path).convert("RGB"), dtype = np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype = np.float32)
        mask = np.stack((mask, mask), 0)'''

        image = nib.load(img_path)
        #print(f"image {image.get_fdata()}")
        image = np.array(image.get_fdata())
        mask = nib.load(mask_path)
        mask = np.array(mask.get_fdata())

        '''mean, std = image.mean((0, 1, 2)), image.std((0, 1, 2))
        transform_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean), (std))
        ])'''

        if self.transform is not None:
            augmentations = self.transform(image = image, mask = mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        #image = np.swapaxes(image, 0, 2)
        '''img_norm = np.array(transform_normalize(image))
        img_norm = np.swapaxes(img_norm, 0, 2)
        img_norm = np.swapaxes(img_norm, 0, 1)
        image = img_norm'''
        print(f"image {img_path} {image.shape} mask {mask_path} {mask.shape}")

        #print(f"image shape {img_path} {image.shape} mask {mask_path} {mask.shape}")
        # target shape (320, 320, 104)

        if image.shape[0] == 256:
            image = np.swapaxes(image, 0, 1)
        if mask.shape[0] == 256:
            mask = np.swapaxes(mask, 0, 1)

        if image.shape[0] == 212:
            #print(f"image path {img_path} shape {image.shape}")
            image = np.pad(image, ((6, 6),(0, 0),(0, 0)), 'constant', constant_values = 0)
            #print(f"image path {img_path} shape {image.shape}")

        if mask.shape[0] == 212:
            #print(f"image path {img_path} shape {image.shape}")
            mask = np.pad(mask, ((6, 6),(0, 0),(0, 0)), 'constant', constant_values = 0)

        return image, mask
