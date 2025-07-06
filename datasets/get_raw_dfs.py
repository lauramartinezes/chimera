import glob
import os
import pandas as pd


def get_df_subset(data_dir, subset, main_insect_class, mislabeled_insect_class):
    sample_configs = [
        {
            'folder': f'{main_insect_class}_good',
            'label': 0,
            'mislabeled': False,
            'measurement_noise': False
        },
        {
            'folder': f'{main_insect_class}_trash',
            'label': 1,
            'mislabeled': False,
            'measurement_noise': True
        },
        {
            'folder': f'{mislabeled_insect_class}_for_{main_insect_class}',
            'label': 1,
            'mislabeled': True,
            'measurement_noise': False
        }
    ]

    df_list = []

    for config in sample_configs:
        folder_path = os.path.join(data_dir, subset, main_insect_class, config['folder'])
        filepaths = glob.glob(os.path.join(folder_path, '*.png'))

        df = pd.DataFrame({
            'label': config['label'],
            'mislabeled': config['mislabeled'],
            'measurement_noise': config['measurement_noise'],
            'filepath': filepaths
        })
        df_list.append(df)

    return pd.concat(df_list, ignore_index=True)


def get_df_test(classes, data_dir):
    df_classes_test = []
    for i, cls in enumerate(classes):
        class_folder_test = os.path.join(data_dir, 'test', cls)
        class_files_test = glob.glob(os.path.join(class_folder_test, '*.png'))

        df_class_test = pd.DataFrame({
            'label': i,
            'mislabeled': False,
            'measurement_noise': False, 
            'filepath': class_files_test
        })
        df_classes_test.append(df_class_test)

    return pd.concat(df_classes_test, ignore_index=True)