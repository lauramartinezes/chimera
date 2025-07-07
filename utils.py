import os
import random
import numpy as np
import torch


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