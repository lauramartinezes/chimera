import os
import shutil
import pandas as pd

from config.config import SAVE_DIR

saved_model = 'fuji_tf_efficientnetv2_m.in21k_ft_in1k_pretrained_imagenet_None_seed_x_augmented'
saved_model_path = f'models/{saved_model}_best.pth.tar'
original_fuji_folder = r'D:\All_sticky_plate_images\created_data\fuji_tile_exports'

setting = 'fuji'

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

    model_i_path = os.path.join(iteration_folder, f'{saved_model}_best_{i}.pth.tar')

    if i == 0:
        shutil.copy(saved_model_path, model_i_path)
        shutil.copy(original_fuji_folder, inlier_folder)

        models.append(model_i_path)
        datasets.append(inlier_folder)
        
        #accs_0 = test(model_0, test_data)

    else:
        # umaps_i, dataset_i = get outliers(models[i-1], datasets[i-1], active_classes)
        if inactive_classes:
            print('')
            # copy the inactive_classes in dataset_i (inliers)
        # model_i = train(models[i-1], dataset[i], val_data, all_classes)
        # accs_i = test(model_i, test_data)
        # active_classes = active_classes[accs_i > accs_i-1]
        # inactive_classes = active_classes[accs_i <= accs_i-1]
        if not active_classes:
            break