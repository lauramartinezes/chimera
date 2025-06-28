import glob
import shutil
from skimage import exposure, io
import numpy as np
import os
from tqdm import tqdm

def ignore_files(dir, files):
    return [file for file in files if os.path.isfile(os.path.join(dir, file))]


train_folder = "data/train_dif_colour"
val_folder = "data/val_dif_colour"
test_folder = "data/test_dif_colour"

# Create new folders
new_train_folder = "data/train"
new_val_folder = "data/val"
new_test_folder = "data/test"

shutil.copytree(train_folder, new_train_folder, dirs_exist_ok=True, ignore=ignore_files)
shutil.copytree(val_folder, new_val_folder, dirs_exist_ok=True, ignore=ignore_files)
shutil.copytree(test_folder, new_test_folder, dirs_exist_ok=True, ignore=ignore_files)

# get list of filepaths using glob
train_filepaths = glob.glob(os.path.join(train_folder, '*', '*','*.png'))
val_filepaths = glob.glob(os.path.join(val_folder, '*', '*','*.png'))
test_filepaths = glob.glob(os.path.join(test_folder, '*','*.png'))

# Merge the two lists
filepaths = train_filepaths + val_filepaths + test_filepaths

reference_wmv = io.imread("data/train_dif_colour/wmv/wmv_good/2020_herent_w28_4-30_4183x6307_14404.png")
reference_c = io.imread("data/train_dif_colour/c/c_good/2020_herent_w32_3-30_4225x6287_85215.png")
reference_meas_noise = io.imread("data/train_dif_colour/wmv/wmv_trash/2019_beauvechain_w27_C_4183x6278_35568.png")

for fp in tqdm(filepaths, desc="Normalizing images"):
    if 'trash' in fp:
        ref = reference_meas_noise
    elif 'wmv_for_c' or 'wmv_good' or 'test_dif_colour/wmv' in fp:
        ref = reference_wmv
    elif 'c_for_wmv' or 'c_good' or 'test_dif_colour/c'in fp:
        ref = reference_c
    img = io.imread(fp)
    matched = exposure.match_histograms(img, ref, channel_axis=-1)
    new_fp = fp.replace('_dif_colour', '')
    io.imsave(new_fp, matched)

print("Images normalized and saved to new folders")
