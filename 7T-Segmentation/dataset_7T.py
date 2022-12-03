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
        self.target_size = (224, 256, 48)

    
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
        matrix = image.affine
        image = np.array(image.get_fdata())
        mask = nib.load(mask_path)
        mask = np.array(mask.get_fdata())

        mean, std = image.mean((0, 1, 2)), image.std((0, 1, 2))
        transform_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean), (std))
        ])

        if self.transform is not None:
            augmentations = self.transform(image = image, mask = mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        #image = np.swapaxes(image, 0, 2)

        #print(f"image {img_path} {image.shape} mask {mask_path} {mask.shape}, target shape {self.target_size} mean {mean} std {std}")

        img_norm = np.array(transform_normalize(image))
        img_norm = np.swapaxes(img_norm, 0, 2)
        img_norm = np.swapaxes(img_norm, 0, 1)
        img_norm = image 

        print(f"image {img_path} {image.shape} mask {mask_path} {mask.shape}, target shape {self.target_size}")

        crop0 = (image.shape[0] - self.target_size[0]) // 2
        crop1 = (image.shape[1] - self.target_size[1]) // 2
        crop2 = (image.shape[2] - self.target_size[2]) // 2
        #print(f"crop {crop0} {crop1} {crop2}")
        
        image = img_norm[crop0:img_norm.shape[0]-crop0, crop1:img_norm.shape[1]-crop1, crop2:img_norm.shape[2]-crop2]
        mask = mask[crop0:mask.shape[0]-crop0, crop1:mask.shape[1]-crop1, crop2:mask.shape[2]-crop2]
        #print(f"image {img_path} {image.shape} mask {mask_path} {mask.shape}, target shape {self.target_size}")
        if image.shape[2] != self.target_size[2] and crop2 > 0:
            residual = image.shape[2] - self.target_size[2]
            image = image[:, :, 0: image.shape[2]-residual]
            mask = mask[:, :, 0: mask.shape[2]-residual]
        elif image.shape[2] != self.target_size[2] and crop2 < 0:
            image = np.pad(image, ((0, 0),(0, 0),(-crop2, -crop2)), 'constant', constant_values = 0)
            mask = np.pad(mask, ((0, 0),(0, 0),(-crop2, -crop2)), 'constant', constant_values = 0)

        #print(f"image {img_path} {image.shape} mask {mask_path} {mask.shape}, target shape {self.target_size}")
        print(f"after image {img_path} {image.shape} mask {mask_path} {mask.shape}, target shape {self.target_size}")

        return image, mask, matrix