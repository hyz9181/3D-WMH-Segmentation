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
#T2_img_path = glob.glob('/mnt/Data_Backup_Cloud/WMH_Segmentation/7T_Data/raw_data/train/')
T2_img_path_short = "/mnt/Data_Backup_Cloud/WMH_Segmentation/7T_Data/raw_data/test/"
#T2_img_path.sort()
#T2_img_path = T2_img_path[:]

def adjust(T2_img_path, normalizer, index):
    print(T2_img_path)
    img = nib.load(T2_img_path).get_fdata()
    affine_mat = nib.load(T2_img_path).affine

    #out_plist = T2_img_path.split('/')[:-1]
    #out_plist.insert(0,'/')
    out_path = T2_img_path_short + str(int(line))
    #out_plist.append(str(index))
    #out_plist += str(int(line))
    out_path += "_nyul_3T.nii.gz"
    #out_path = os.path.join(*out_plist)
    out_img = normalizer(img)
    
    nii_img = nib.Nifti1Image(out_img, affine=affine_mat)
    print(f"out path {out_path}")
    nib.save(nii_img, out_path)

nyul_normalizer = NyulNormalize()
nyul_normalizer.load_standard_histogram("3T_standard_histogram.npy")
fd = open('test.txt', 'r')
#content = fd.read()
line = fd.readline()

while line:
    T2_img_path = T2_img_path_short + str(int(line))
    T2_img_path += ".nii.gz"
    #print(T2_img_path)
    adjust(T2_img_path, nyul_normalizer, line)
    line = fd.readline()

#for i in tqdm(range(len(T2_img_path))):
    
