import glob
import os

import pandas as pd

from torchvision import datasets, transforms

good_samples_folder = os.path.join('data', 'wmv_good')
mismeasure_samples_folder = os.path.join('data', 'wmv_trash')
mislabel_samples_folder = os.path.join('data', 'sp_good')

good_samples = glob.glob(os.path.join(good_samples_folder, '*.png'))
mismeasure_samples = glob.glob(os.path.join(mismeasure_samples_folder, '*.png'))
mislabel_samples = glob.glob(os.path.join(mislabel_samples_folder, '*.png'))

df_good_samples = pd.DataFrame({
    'label': 0,
    'noise_label': 0,
    'filepath': good_samples
})

df_mismeasure_samples = pd.DataFrame({
    'label': 1,
    'noise_label': 1,
    'filepath': mismeasure_samples
})

df_mislabel_samples = pd.DataFrame({
    'label': 1,
    'noise_label': 2,
    'filepath': mislabel_samples
})

df_all_samples = pd.concat([df_good_samples, df_mismeasure_samples, df_mislabel_samples], ignore_index=True)

batch_size = 1024
transform_resnet = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a tensor
    # transforms.Resize(28),  # Resize the image to 28x28
])