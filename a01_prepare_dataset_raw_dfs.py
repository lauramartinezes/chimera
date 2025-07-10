import os

import yaml

from datasets.get_raw_dfs import get_df_subset, get_df_test


if __name__ == '__main__':
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    insect_classes = config["data_params"]["data_classes"]
    data_dir = config["data_params"]["splitted_data_dir"]
    subsets = ['train', 'val']
    
    for i, main_insect_class in enumerate(insect_classes):
        mislabeled_insect_class = insect_classes[1 - i]
        
        for subset in subsets:
            df_subset_path = os.path.join(
                data_dir, 
                f'df_{subset}_{config["data_params"]["raw_suffix"]}_{main_insect_class}.csv'
            )
            if not os.path.exists(df_subset_path):
                df_subset = get_df_subset(data_dir, subset, main_insect_class, mislabeled_insect_class)
                df_subset.to_csv(df_subset_path)
    
    df_test_path = os.path.join(data_dir, f'df_test.csv')
    if not os.path.exists(df_test_path):
        df_test = get_df_test(insect_classes, data_dir)
        df_test.to_csv(df_test_path)
