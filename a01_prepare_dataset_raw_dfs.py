import os

from datasets.get_raw_dfs import get_df_subset, get_df_test
from utils import load_config


if __name__ == '__main__':
    config = load_config("config.yaml")

    insect_classes = config["data_params"]["data_classes"]
    data_dir = config["data_params"]["splitted_data_dir"]
    files_extension = config["data_params"]["file_extension"]
    subsets = ['train', 'val']
    
    for i, main_insect_class in enumerate(insect_classes):
        mislabeled_insect_class = insect_classes[1 - i]
        
        for subset in subsets:
            df_subset_path = os.path.join(
                data_dir, 
                f'df_{subset}_{config["data_params"]["raw_suffix"]}_{main_insect_class}.csv'
            )
            if not os.path.exists(df_subset_path):
                print(f"Creating DataFrame for {subset} subset of {main_insect_class}...")
                df_subset = get_df_subset(data_dir, subset, main_insect_class, mislabeled_insect_class, files_extension=files_extension)
                df_subset.to_csv(df_subset_path)
            else:
                print(f"DataFrame for {subset} subset of {main_insect_class} already exists at {df_subset_path}. Skipping creation.")
    
    df_test_path = os.path.join(data_dir, f'df_test.csv')
    if not os.path.exists(df_test_path):
        print("Creating DataFrame for test subset...")
        df_test = get_df_test(insect_classes, data_dir, files_extension=files_extension)
        df_test.to_csv(df_test_path)
    else:
        print(f"DataFrame for test subset already exists at {df_test_path}. Skipping creation.")
