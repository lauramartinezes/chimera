# CHIMERA: A Pipeline for Cleaning Noisy Image Datasets
## 📚 Paper 
[![Paper](https://img.shields.io/badge/Paper-SSRN-blue)](https://dx.doi.org/10.2139/ssrn.6169047)

If you would like to cite this work or read the full details, please see:

**CHIMERA: Cleaning Heterogeneous Image Datasets from Measurement and Label Noise for Robust Classification Accuracy**  
📄 [https://dx.doi.org/10.2139/ssrn.6169047](https://dx.doi.org/10.2139/ssrn.6169047)


## 🔍 What is CHIMERA?

CHIMERA is a pipeline for cleaning image datasets by separating:
- **Measurement noise** (bad images, artifacts)
- **Label noise** (mislabelled images)

It helps improve dataset quality and model performance with minimal manual effort.

## 💡 Getting Started
CHIMERA supports multiple dataset workflows. Choose the one that fits your case:

- [Run with FashionMNIST Dataset](#a-fashionmnist-dataset)  
- [Run with Insect Dataset](#b-insect-dataset)  
- [Run with a Custom Dataset](#c-custom-dataset)

> Tip: If you are unsure, start with FashionMNIST as it is the quickest way to test the pipeline.


## 📦 Installation
### 1. Create a conda environment
```
conda create -n chimera python=3.10.8
conda activate chimera
```

### 2. Install Pytorch (with GPU) 
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
If this fails, follow the instructions on the official [PyTorch website](https://pytorch.org/get-started/locally/).

### 3. Install additional dependencies
```
pip install -r requirements.txt
```

## 🚀 Usage
### A) FashionMNIST Dataset
Run the bash script `run_pipeline.sh` from terminal
```
bash run_pipeline.sh
```

Results will be stored in the `logs` folder
### B) Insect Dataset
1. Contact wouter.saeys@kuleuven.be to get access to the `phoneboxdata` and the `split_60_20_20` folders
    - [`phoneboxdata`](https://kuleuven.sharepoint.com/:f:/r/sites/T0006791/Shared%20Documents/LA%20Insects/PhD%20-%20Laura%20Martinez%20Esmeral/WP2/phoneboxdata?csf=1&web=1&e=dFP3tJ)
   - [`split_60_20_20`](https://kuleuven.sharepoint.com/:f:/r/sites/T0006791/Shared%20Documents/LA%20Insects/PhD%20-%20Laura%20Martinez%20Esmeral/WP2/split_60_20_20?csf=1&web=1&e=to8rQM)
2. Place both folders inside the `data/` directory
3. Rename `config_phonebox.yaml` to `config.yaml`
4. Run:
    ```
    bash run_pipeline.sh
    ```

### C)  Custom Dataset
1. Organize your dataset:
    ```python
    data/
    your_dataset/
        class_a/
        class_b/
        class_trash/ # this will be your measurement noise
        ...
    ```
2. Modify inside `config.yaml` the `data_params`:
    - `file_extension`
    - `original_data_dir`
    - `data_classes` (classes of interest)
    - `trash_class` (measurement noise)

3. Run:
    ```
    bash run_pipeline.sh
    ```
### 📊 Visualization (Optional)

To explore UMAP projections interactively:
```
python a05_visualize_umaps_in_app.py
```