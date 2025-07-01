import random
import numpy as np
import torch

from sklearn.metrics import roc_auc_score, average_precision_score


def set_seed(seed):
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
    # os.environ['TF_DETERMINISTIC_OPS'] = 'true'

    # basic seed
    np.random.seed(seed)
    random.seed(seed)

    # tensorflow seed
    # try:
    #     tf.random.set_seed(seed) # for tf >= 2.0
    # except:
    #     tf.set_random_seed(seed)
    #     tf.random.set_random_seed(seed)

    # pytorch seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def metric(y_true, y_score, pos_label=1):
    aucroc = roc_auc_score(y_true=y_true, y_score=y_score)
    aucpr = average_precision_score(y_true=y_true, y_score=y_score, pos_label=pos_label)

    return {'aucroc':aucroc, 'aucpr':aucpr}