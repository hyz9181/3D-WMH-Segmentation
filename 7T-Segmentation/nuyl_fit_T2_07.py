import nibabel as nib
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tqdm import tqdm
import pdb
import psutil
from intensity_normalization.normalize.nyul import NyulNormalize


#T2_img_path = glob.glob('/mnt/Mercury/ADNI_2020/Public/Analysis/data/ADNI1/ADNI1_data/*/*/converted/Double_TSE/regis*.nii.gz')
T2_img_path = glob.glob('/mnt/Data_Backup_Cloud/WMH_Segmentation/wmh_data/new_data/T2_FLAIR/*.nii')
T2_img_path.sort()
T2_img_path = T2_img_path[:]
T2_images = []
T2_images_affine = []

for i in tqdm(range(len(T2_img_path))):
    
    T2_images.append(nib.load(T2_img_path[i]).get_fdata())
    T2_images_affine.append(nib.load(T2_img_path[i]).affine)
    if i % 10 == 0:
        print(f'RAM memory % used: {psutil.virtual_memory()[2]}')
        
    if psutil.virtual_memory()[2] > 95:
        break

nyul_normalizer = NyulNormalize()
nyul_normalizer.fit(T2_images)
nyul_normalizer.save_standard_histogram("3T_standard_histogram.npy")