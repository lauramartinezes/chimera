import glob
import os
import random

import numpy as np
import pandas as pd
import torch

def get_df_train(main_insect_class, mislabeled_insect_class):
    good_samples_folder_train = os.path.join('data', 'train', main_insect_class, f'{main_insect_class}_good')
    mismeasure_samples_folder_train = os.path.join('data', 'train', main_insect_class, f'{main_insect_class}_trash')
    mislabel_samples_folder_train = os.path.join('data', 'train', main_insect_class, f'{mislabeled_insect_class}_for_{main_insect_class}')

    good_samples_train = glob.glob(os.path.join(good_samples_folder_train, '*.png'))
    mismeasure_samples_train = glob.glob(os.path.join(mismeasure_samples_folder_train, '*.png'))
    mislabel_samples_train = glob.glob(os.path.join(mislabel_samples_folder_train, '*.png'))

    df_good_samples_train = pd.DataFrame({
        'label': 0,
        'mislabeled': False,
        'measurement_noise': False, 
        'filepath': good_samples_train
    })

    df_mismeasure_samples_train = pd.DataFrame({
        'label': 1,
        'mislabeled': False,
        'measurement_noise': True, 
        'filepath': mismeasure_samples_train
    })

    df_mislabel_samples_train = pd.DataFrame({
        'label': 1,
        'mislabeled': True,
        'measurement_noise': False, 
        'filepath': mislabel_samples_train
    })

    df_train = pd.concat([df_good_samples_train, df_mismeasure_samples_train, df_mislabel_samples_train], ignore_index=True)

    return df_train


def get_df_test(class_1, class_2):
    class_1_folder_test = os.path.join('data', 'test', class_1)
    class_2_folder_test = os.path.join('data', 'test', class_2)

    class_1_test = glob.glob(os.path.join(class_1_folder_test, '*.png'))
    class_2_test = glob.glob(os.path.join(class_2_folder_test, '*.png'))

    df_class_1_test = pd.DataFrame({
        'label': 0,
        'mislabeled': False,
        'measurement_noise': False, 
        'filepath': class_1_test
    })

    df_class_2_test = pd.DataFrame({
        'label': 1,
        'mislabeled': False,
        'measurement_noise': False, 
        'filepath': class_2_test
    })

    df_test = pd.concat([df_class_1_test, df_class_2_test], ignore_index=True)
    return df_test


if __name__ == '__main__':
    # Set manual seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    insect_classes = ['wmv', 'c']
    
    for i in range(len(insect_classes)):
        main_insect_class = insect_classes[i]
        mislabeled_insect_class = insect_classes[1 - i]
        
        df_train_path = os.path.join('data', f'df_train_ae_{main_insect_class}.csv')
        if not os.path.exists(df_train_path):
            df_train = get_df_train(main_insect_class, mislabeled_insect_class)
            df_train.to_csv(df_train_path)
    
    df_test_path = os.path.join('data', f'df_test.csv')
    if not os.path.exists(df_test_path):
        df_test = get_df_test(insect_classes[0], insect_classes[1])
        df_test.to_csv(df_test_path)
