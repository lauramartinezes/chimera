#!/bin/bash

files_with_args=(
  "a00_split_dataset_raw_dfs.py"
  "a01_prepare_dataset_raw_dfs.py"
  "a02_train_test_classifier.py"
  "a03_prepare_data_swap_dfs.py"
  "a04_generate_umaps.py"
  "a06_evaluate_outlier_detection.py"
  "a07_clean_dfs.py"
  "a08_analyze_meas_label_noise.py"
  "a02_train_test_classifier.py --experiments all_cases"
)

for entry in "${files_with_args[@]}"; do
    file="${entry%% *}"
    args="${entry#"$file"}"

    echo "Running $file$args"
    python "$file" $args 2> >(grep -v '^\s*$' | grep -vi 'warning' >&2)

    if [ $? -ne 0 ]; then
        echo "$file failed. Exiting."
        exit 1
    fi
done

echo "All scripts executed successfully!"
