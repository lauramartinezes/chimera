# stickybugs-outliers


## Abstract
[Insert abstract here]

## Installation
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

## How-to
### Main steps

1. Create a `data` folder inside this repo. 

2. Add the following folders to `data`:
   - [`phoneboxdata`](https://kuleuven.sharepoint.com/:f:/r/sites/T0006791/Shared%20Documents/LA%20Insects/PhD%20-%20Laura%20Martinez%20Esmeral/WP2/phoneboxdata?csf=1&web=1&e=dFP3tJ)
   - [`split_60_20_20`](https://kuleuven.sharepoint.com/:f:/r/sites/T0006791/Shared%20Documents/LA%20Insects/PhD%20-%20Laura%20Martinez%20Esmeral/WP2/split_60_20_20?csf=1&web=1&e=to8rQM)


3. Run the bash script `run_pipeline.sh` from terminal
    ```
    bash run_pipeline.sh
    ```

4. Check the  results stored in the `logs` folder

### *(Optional)*

5. To look at the umap projections in an interactive manner, run `a05_visualize_umaps_in_app.py`

5. To try other datasets, splitting proportions or other configurations, update `config.yaml` and repeat step 3 and 4