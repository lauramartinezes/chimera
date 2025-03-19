#!/bin/bash

# Execute the first script
# echo "Running a02_1_train_adversarial_ae.py"
# python "a02_1_train_adversarial_ae.py"
# if [ $? -ne 0 ]; then
#     echo "a02_1_train_adversarial_ae.py failed. Exiting."
#     exit 1
# fi

# echo "Running a02_2_train_ae.py"
# python "a02_2_train_ae.py"
# if [ $? -ne 0 ]; then
#     echo "a02_2_train_ae.py failed. Exiting."
#     exit 1
# fi

# echo "Running a03_visualize_reconstructions_ae.py"
# python "a03_visualize_reconstructions_ae.py"
# if [ $? -ne 0 ]; then
#     echo "a03_visualize_reconstructions_ae.py failed. Exiting."
#     exit 1
# fi

# echo "Running a04_1_umap_projections_ae.py"
# python "a04_1_umap_projections_ae.py"
# if [ $? -ne 0 ]; then
#     echo "a04_1_umap_projections_ae.py failed. Exiting."
#     exit 1
# fi

# echo "Running a04_2_umap_projections_cnn.py"
# python "a04_2_umap_projections_cnn.py"
# if [ $? -ne 0 ]; then
#     echo "a04_2_umap_projections_cnn.py failed. Exiting."
#     exit 1
# fi

# echo "Running a06_outliers_evaluation.py"
# python "a06_outliers_evaluation.py"
# if [ $? -ne 0 ]; then
#     echo "a06_outliers_evaluation.py failed. Exiting."
#     exit 1
# fi

echo "Running a07_clean_data.py"
python "a07_clean_data.py"
if [ $? -ne 0 ]; then
    echo "a07_clean_data.py failed. Exiting."
    exit 1
fi

# echo "Running a08_visualize_resnet_outlier_detection.py"
# python "a08_visualize_resnet_outlier_detection.py"
# if [ $? -ne 0 ]; then
#     echo "a08_visualize_resnet_outlier_detection.py failed. Exiting."
#     exit 1
# fi

echo "Running a01_train_test_classifier.py"
python "a01_train_test_classifier.py"
if [ $? -ne 0 ]; then
    echo "a01_train_test_classifier.py failed. Exiting."
    exit 1
fi

echo "All scripts executed successfully!"
