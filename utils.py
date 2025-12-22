import os
import random
import numpy as np
import torch
import yaml


def save_config(config, file_path):
    with open(file_path, "w") as f:
        yaml.dump(config, f)


def load_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def set_seed(seed):
    # basic seed
    np.random.seed(seed)
    random.seed(seed)

    # pytorch seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_dir_empty(path):
    if not os.path.isdir(path):
        return True  
    return not any(os.scandir(path))