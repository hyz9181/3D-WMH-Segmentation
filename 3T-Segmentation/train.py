import os
import sys
import numpy as np
import nibabel as nib
import torch
import torchvision
from torch import nn
from torch import squeeze
import torch.optim as optim
import torch.nn.functional as F
#from monai.networks.nets import UNETR
from monai.losses import DiceCELoss, DiceLoss
#from unetr import UNETR
from UNET3 import UNET
from dataset import WMH_Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
#torch.cuda.empty_cache()
#torch.cuda.memory_summary(device=None, abbreviated=False)
LEARNING_RATE = 1e-3
Batch_size = 1
NUM_EPOCH = 10
DEVICE = device
writer = SummaryWriter("runs/3DUNet_3TData")

Train_Img_Dir = "/mnt/Data_Backup_Cloud/WMH_Segmentation/wmh_data/new_data/T2_FLAIR/"  # Flair Training Image Directory
#Train_Img_Dir = "/mnt/Data_Backup_Cloud/wmh_data/new_data/T2/T2/"  # T2 Training Image Directory
#Train_Img_Dir = "/mnt/Data_Backup_Cloud/wmh_data/new_data/T1/"  # T1 Training Image Directory
Train_Mask_Dir = "/mnt/Data_Backup_Cloud/WMH_Segmentation/wmh_data/new_data/Mask/FLAIR_mask/"  # Mask Image Directory

Test_Img_Dir = "/mnt/Data_Backup_Cloud/WMH_Segmentation/wmh_data/new_data/FLAIR_unseen/"  # Flair Test Image Directory
#Test_Img_Dir = "/mnt/Data_Backup_Cloud/wmh_data/new_data/T2_unseen/"  # T2 Test Image Directory
#Test_Img_Dir = "/mnt/Data_Backup_Cloud/wmh_data/new_data/T1_unseen/"  # T1 Test Image Directory
Test_Mask_Dir = "/mnt/Data_Backup_Cloud/WMH_Segmentation/wmh_data/new_data/Mask/unseen_mask_FLAIR/"  # Test Mask Directory
save_dir = "/mnt/Data_Backup_Cloud/WMH_Segmentation/3T_result/"

#Train_Img_Dir = "/mnt/Data_Backup_Cloud/wmh_data/new_data/T2/T2/"
#Train_Mask_Dir = "/mnt/Data_Backup_Cloud/wmh_data/new_data/T2/mask/"


def get_loaders(train_dir, 
                train_mask_dir, 
                batch_size,
                train_transform,
                ):
    full_dataset = WMH_Dataset(image_dir = train_dir, mask_dir = train_mask_dir, transform = train_transform)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    return train_loader, val_loader

def get_test_loaders(test_dir, test_mask_dir, batch_size, test_transform):
    test_dataset = WMH_Dataset(image_dir = test_dir, mask_dir = test_mask_dir, transform = test_transform)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True,)
    return test_loader

def IoULoss(inputs, targets, smooth = 1e-6):
    #comment out if your model contains a sigmoid or equivalent activation layer\
    inputs = torch.sigmoid(inputs)       
        
    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
        
    #intersection is equivalent to True Positive count
    #union is the mutually inclusive area of all labels & predictions 
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 
        
    IoU = (intersection + smooth)/(union + smooth)
    #print(f"IoU Loss {IoU}")
    return 1 - IoU

def cut_patches(data, patch_size):
    x_npatch = data.shape[0] // patch_size
    y_npatch = data.shape[1] // patch_size
    z_npatch = data.shape[2] // patch_size
    prev_x, prev_y, prev_z = 0

    data_cut = []
    for i in range(x_npatch):
        for j in range(y_npatch):
            for k in range(z_npatch):
                data[prev_x]



def train(loader, model, optimizer, loss_fn, scaler, epochs):
    model.train()
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    train_loss = 0.0
    train_accuracy = 0

    for batch_idx, (data, targets) in enumerate(loader):

        patch_size = 16

        data = data.float().unsqueeze(1).to(device = DEVICE)
        targets = targets.float().unsqueeze(1).to(device = DEVICE)
        
        pred = data

        with torch.cuda.amp.autocast():
            predictions = model(data)
            #print(f"predictions shape {predictions.shape}")
            pred = predictions
            print(f"pred {predictions.shape}")
            loss = IoULoss(predictions, targets)
        
        #print(f"test batch index {batch_idx}")
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        preds = torch.sigmoid(pred)
        preds = (preds > 0.5).float()
        num_correct += (torch.eq(preds, targets)).sum()
        num_pixels += torch.numel(preds)
        print(f"loss: {loss} with batch index {batch_idx}")

        writer.add_scalar('training loss', loss, epochs * 76 + batch_idx)
        writer.add_scalar('train accuracy', num_correct/num_pixels*100, epochs * 76 + batch_idx)

        #print(f"num correct {num_correct} num pixels {num_pixels} pred {preds.shape} targets {targets.shape} pred targets equal {(preds == targets).shape}")
        dice_score += (2 * (preds * targets).sum()) / ((preds + targets).sum() + 1e-8)
        print(
            f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100}"
        )
        print(f"Dice score: {dice_score/len(loader)}")
        writer.add_scalar('train dice score', dice_score, epochs * 76 + batch_idx)
        num_correct = 0
        num_pixels = 0
        dice_score = 0

        print()

def check_accuracy(loader, model, epochs, device = "cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.float().unsqueeze(1).to(device)
            y = y.unsqueeze(1).to(device)
            #print(f"x shape {x.shape} y shape {y.shape}")
            preds = torch.sigmoid(model(x))
            loss = IoULoss(preds, y)
            preds = (preds > 0.5).float()
            num_correct += (torch.eq(preds, y)).sum()
            num_pixels += torch.numel(preds)
            print(f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100} batch idx {batch_idx}")
            print()
            writer.add_scalar('validation loss', loss, epochs * 20 + batch_idx)
            writer.add_scalar('validation accuracy', num_correct/num_pixels*100, epochs * 20 + batch_idx)

            num_correct = 0
            num_pixels = 0
            dice_score = 0



def test(loader, model, device = "cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.unsqueeze(1).float().to(device)
            #x = x.squeeze(1)
            y = y.unsqueeze(1).to(device)
            #print(f"x shape {x.shape} y shape {y.shape}")
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            print(f"Test batch index {batch_idx} Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100}")

            x_numpy = (x.cpu().detach().numpy())[0, 0, :, :, :]
            pred_numpy = np.flip((preds.cpu().detach().numpy())[0, 0, :, :, :], axis = 0)
            y_numpy = np.flip((y.cpu().detach().numpy())[0, 0, :, :, :], axis = 0)

            if x_numpy.shape[0] == 212:
                x_numpy = np.pad(x_numpy, ((6, 6),(0, 0),(0, 0)), 'constant', constant_values = 0)
                    
            x_img = nib.Nifti1Image(x_numpy, affine = np.eye(4))
            pred_img = nib.Nifti1Image(pred_numpy, affine = np.eye(4))
            y_img = nib.Nifti1Image(y_numpy, affine = np.eye(4))
            nib.save(x_img, save_dir + str(batch_idx) + 'test_input_T1.nii')
            nib.save(pred_img, save_dir + str(batch_idx) + 'val_pred_T1.nii')
            nib.save(y_img, save_dir + str(batch_idx) + 'g_truth_T1.nii')
            
            num_correct = 0
            num_pixels = 0
            dice_score = 0


#model = UNET(1, 1)
model = UNET(1, 1).to(DEVICE)
model = model.to(DEVICE)
summary(model, input_size = (224, 256, 48))

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.2)),
])

train_loader, val_loader = get_loaders(
    Train_Img_Dir,
    Train_Mask_Dir,
    Batch_size,
    train_transform = None, 
)

test_loader = get_test_loaders(Test_Img_Dir, Test_Mask_Dir, Batch_size, test_transform = None,)

print(f"start training ...")

scaler = torch.cuda.amp.GradScaler()
for epochs in range(NUM_EPOCH):
    print(f"\nEpoch {epochs+1}\n -------------------------------")
    print("Training...")
    train(train_loader, model, optimizer, loss_fn, scaler, epochs)
    print()
    check_accuracy(val_loader, model, epochs, device = DEVICE)
writer.flush()
print("Training Done!\n")
test(test_loader, model, device = DEVICE)
torch.save(model, "3d_unet.pth")
writer.close()
