# CHIMERA: Cleaning Heterogeneous Image Datasets from Measurement and Label Noise for Robust Classification Accuracy


## Abstract
AI model performance relies on high-quality data, yet in agriculture, collecting large, clean datasets is challenging due to time-consuming acquisition and biological variability, making data cleaning fundamental. Traditional cleaning often focuses on measurement noise, corrupted or irrelevant inputs like artifacts or non-target objects that deviate from the valid data distribution and are therefore treated as outliers. Equally important is label noise, which arises when valid inputs are assigned labels inconsistent with their feature representations, manifesting as anomalies in the feature–label relationship rather than as statistical outliers. Although both noise types have been widely studied, they are typically handled independently, making a unified strategy essential. We propose CHIMERA (Cleaning Heterogeneous Image Datasets from Measurement and Label Noise for Robust Classification Accuracy), a framework that detects and separates measurement and label noise in image classification, discarding the former and flagging the latter for relabelling. CHIMERA begins by fine-tuning a pretrained network on the noisy samples to generate predicted labels, assumed to better reflect ground truth. Based on these labels, samples are grouped and processed via feature extraction and outlier detection. Detected outliers are deemed measurement noise, while inliers with mismatched predicted and original labels are flagged as label noise. Using an insect classification task for pest monitoring, we demonstrate how CHIMERA improves dataset quality and classification performance, achieving 90.52% accuracy, compared to 87.93% on the noisy dataset. By disentangling measurement and label noise, CHIMERA provides a practical approach to clean image datasets, enhancing the robustness of models in resource-constrained agricultural settings. 

## 1. Installation
Create a conda environment and activate it
```
conda create -n stickyod python=3.10.8
conda activate stickyod
```

Add pytorch library (with GPU) to your environment (if the command below does not work follow the instructions in https://pytorch.org/get-started/locally/)
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

Next, install pip and the jupyter notebook related libraries
```
conda install pip jupyterlab ipython ipywidgets nb_conda_kernels ipykernel
```

and, to install the remaining libraries, run 
```
pip install -r requirements.txt
```

## 2. How-to
### 2.1 Using the FashionMNIST Dataset
1. Run the bash script `run_pipeline.sh` from terminal
    ```
    bash run_pipeline.sh
    ```

2. Check the  results stored in the `logs` folder
### 2.2 Using the Phonebox Dataset
1. Contact wouter.saeys@kuleuven.be to get access to the `phoneboxdata` and the `split_60_20_20` folders
    - [`phoneboxdata`](https://kuleuven.sharepoint.com/:f:/r/sites/T0006791/Shared%20Documents/LA%20Insects/PhD%20-%20Laura%20Martinez%20Esmeral/WP2/phoneboxdata?csf=1&web=1&e=dFP3tJ)
   - [`split_60_20_20`](https://kuleuven.sharepoint.com/:f:/r/sites/T0006791/Shared%20Documents/LA%20Insects/PhD%20-%20Laura%20Martinez%20Esmeral/WP2/split_60_20_20?csf=1&web=1&e=to8rQM)
2. Add the two folders to the `data` directory
3. Rename `config_phonebox.yaml` to `config.yaml`
4. Repeat steps 1 and 2 from Section 2.1

### 2.3 Using a Custom Dataset
1. Organize the dataset into a directory with subfolders for each class
2. Place your dataset inside the `data` directory 
3. Modify the `config.yaml` to specify the dataset folder path, the names of the classes of interest, the name of the mesaurement noise class and the file extension
4. Repeat steps 1 and 2 from Section 2.1

### *(Optional)*

To look at the umap projections in an interactive manner, run `a05_visualize_umaps_in_app.py`