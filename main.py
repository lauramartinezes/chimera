import glob
import os
import shutil
import pandas as pd

from config.config import SAVE_DIR
from detect_outliers import detect_outliers
from train import train

saved_model = 'fuji_tf_efficientnetv2_m.in21k_ft_in1k_pretrained_imagenet_None_seed_x_augmented'
saved_model_path = f'models/{saved_model}_best.pth.tar'
original_fuji_folder = r'D:\All_sticky_plate_images\created_data\fuji_tile_exports'

setting = 'fuji'
n_components = 2

# Read images that were used to train model
df_sticky_dataset_train_big = pd.read_parquet(f"{SAVE_DIR}/df_train_{setting}.parquet")
df_sticky_dataset_val_big = pd.read_parquet(f"{SAVE_DIR}/df_val_{setting}.parquet")
df_sticky_dataset_test_big = pd.read_parquet(f"{SAVE_DIR}/df_test_{setting}.parquet")


iterations = list(range(101))
models = []
datasets = []

all_classes = df_sticky_dataset_train_big['txt_label'].unique().tolist()
active_classes = all_classes.copy()
inactive_classes = []


for i in iterations:
    iteration_folder = os.path.join('output', f'iteration_{i}')
    umaps_folder = os.path.join(iteration_folder, 'umaps')
    solution_outlier_detector_folder = os.path.join(iteration_folder, 'outlier_detections')
    outlier_folder = os.path.join(iteration_folder, 'output_dataset', 'outliers')
    inlier_folder = os.path.join(iteration_folder, 'output_dataset', 'inliers')

    os.makedirs(umaps_folder, exist_ok=True)
    os.makedirs(solution_outlier_detector_folder, exist_ok=True)
    os.makedirs(outlier_folder, exist_ok=True)
    os.makedirs(inlier_folder, exist_ok=True)

    datasets.append(inlier_folder)

    if i == 0:
        model_0_path = os.path.join(iteration_folder, f'{saved_model}_{i}_best.pth.tar')
        models.append(model_0_path)
        shutil.copy(saved_model_path, model_0_path)
        shutil.copy(original_fuji_folder, inlier_folder)
        #accs_0 = test(model_0, test_data)

    else:
        # Separate outliers and inliers
        for species in active_classes:
            if species!='wmv': 
                continue

            # df_sticky_dataset_single_species = df_sticky_dataset_train_big[df_sticky_dataset_train_big.txt_label == species]
            # sticky_dataset_train_filepaths = df_sticky_dataset_single_species.filename.to_list()

            # original_fuji_species = glob.glob(os.path.join(original_fuji_folder, '*', '*'))
            # sticky_dataset_train_filepaths = [path for path in original_fuji_species if os.path.basename(path) not in base_names_val and os.path.basename(path) not in base_names_test]
            # print(len(original_fuji_species), len(sticky_dataset_train_filepaths))

            dataset_species_paths = glob.glob(os.path.join(datasets[i-1], species, '*'))

            outlier_filepaths, inlier_filepaths = detect_outliers(
                    species,
                    dataset_species_paths,
                    active_classes, 
                    all_classes,
                    models[i-1], 
                    iteration_folder, 
                    umaps_folder, 
                    solution_outlier_detector_folder, 
                    outlier_folder, 
                    inlier_folder, 
                    n_components=2
            )
        if inactive_classes:
            for species in inactive_classes:
                species_file_path_i_minus_1 = os.path.join(datasets[i-1], species)
                species_file_path_i = os.path.join(datasets[i], species)
                shutil.copy(species_file_path_i_minus_1, species_file_path_i)
        
        # Train new model
        # Extract basenames from the list of file paths
        # List all files in the folder matching the pattern
        train_i_filepaths = glob.glob(os.path.join(datasets[i], '*', '*'))
        train_i_basenames = [os.path.basename(path) for path in train_i_filepaths]

        # Filter the DataFrame to include only rows where the basename matches
        df_train_i = df_sticky_dataset_train_big[df_sticky_dataset_train_big["filename"].apply(os.path.basename).isin(train_i_basenames)]
        # Train the model i, we dont return it but we could eventually do so, tbd, also we could give all_classes as input
        model_i_path = train(df_train_i, df_sticky_dataset_val_big, df_sticky_dataset_test_big, i, iteration_folder)
        models.append(model_i_path)

        # accs_i = test(model_i, test_data)
        # active_classes = active_classes[accs_i > accs_i-1]
        # inactive_classes = active_classes[accs_i <= accs_i-1]
        if not active_classes:
            break