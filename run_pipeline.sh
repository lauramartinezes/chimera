#!/bin/bash

files=(
  "a00_split_dataset_raw_dfs.py"
  "a01_prepare_dataset_raw_dfs.py"
  "a02_train_test_classifier.py"
  "a03_prepare_data_swap_dfs.py" 
  "a04_generate_umaps.py"
  "a05_visualize_umaps_in_app.py"
  "a06_evaluate_outlier_detection.py"
  "a07_clean_dfs.py" 
  "a08_analyze_meas_label_noise.py" 
)

for file in "${files[@]}"; do
    echo "Running $file"
    python "$file"
    if [ $? -ne 0 ]; then
        echo "$file failed. Exiting."
        exit 1
    fi
done

echo "Running a02_train_test_classifier.py for all cases"
python "a02_train_test_classifier.py" --experiments all_cases
if [ $? -ne 0 ]; then
    echo "a02_train_test_classifier.py failed. Exiting."
    exit 1
fi

echo "All scripts executed successfully!"
