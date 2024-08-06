from collections import OrderedDict
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from config.config import *


basic_df_columns = ['filename', 'label', 'imgname',
                    'platename', 'year', 'location', 'date', 'xtra', 'plate_idx']


def read_image(filename, plot=False):
    img = Image.open(filename)
    return img


def get_files(directory, ext='.jpg'):
    return pd.Series(Path(directory).rglob(f"**/*{ext}"))


def to_weeknr(date=''):
    """
    Transforms a date strings YYYYMMDD to the corresponding week nr (e.g. 20200713 becomes w29)
    """
    week_nr = pd.to_datetime(date).to_pydatetime().isocalendar()[1]
    return f"w{week_nr}"


def extract_filename_info(filename: str, setting='fuji') -> str:
    import os

    if not isinstance(filename, str):
        raise TypeError("Provide the filename as a string.")

    path = Path(filename)
    datadir_len = len(Path(os.path.join(DATA_DIR, setting)).parts)
    parts = path.parts
    label = parts[datadir_len]
    imgname = parts[datadir_len+1]

    if setting == 'fuji' or setting == 'photobox' or 'phoneboxS20FE':
        platename = "_".join(imgname.split('_')[0:-2])
        name_split_parts = imgname.split('_')
        year = name_split_parts[0]
        location = name_split_parts[1]
        date = name_split_parts[2]
        xtra = name_split_parts[3]
        plate_idx = name_split_parts[-1].split('.')[0]

    else:
        raise ValueError()

    return filename, label, imgname[:-4], platename, year, location, date, xtra, plate_idx


def plot_torch_img(x, idx):
    import matplotlib.pyplot as plt
    plt.imshow(x[idx].permute(1, 2, 0))

# def copy_list_of_files(files):


def detect_outliers(X_train, algorithm='KNN'):
    from pyod.models.knn import KNN  # kNN detector
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    if algorithm == 'KNN':
        # train kNN detector
        clf = KNN()
        clf.fit(X_train)

        # get the prediction label and outlier scores of the training data
        # binary labels (0: inliers, 1: outliers)
        return clf.labels_, clf.decision_scores_
    else:
        raise NotImplementedError()


def save_checkpoint(state, is_best, filename='', folder = SAVE_DIR):
    from shutil import copyfile

    import torch
    filename = os.path.join(folder, f'{filename}.pth.tar')
    torch.save(state, filename)
    if is_best:
        copyfile(filename, f"{filename[:-len('.pth.tar')]}_best.pth.tar")


def load_checkpoint(filename, model, optimizer):
    import torch
    assert isinstance(filename, str) and filename.endswith(
        'pth.tar'), "Only works with a pth.tar file."
    checkpoint = torch.load(filename)
    state_dict = checkpoint['state_dict']
    if 'inaturalist' in filename:
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()

        for key, value in state_dict.items():
            if key.startswith("classifier"):
                key = key.replace("classifier.1", "classifier")
            new_state_dict[key] = value
        state_dict = new_state_dict
        
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def copy_files(filelist, destination):
    from shutil import copy2
    for f in tqdm(filelist, total=len(filelist), desc="Copying files.."):
        copy2(f, destination)


@torch.no_grad()
def get_all_preds(model, loader, dataframe=False, final_nodes=2):
    import torch
    all_preds = torch.tensor([]).cuda()
    all_labels = torch.tensor([]).cuda()
    all_paths = []
    all_idx = torch.tensor([]).cuda()
    for x_batch, y_batch, path_batch, idx_batch in tqdm(loader):

        preds = model(x_batch.cuda())
        all_preds = torch.cat((all_preds, preds), dim=0)
        all_labels = torch.cat((all_labels, y_batch.cuda()), dim=0)
        all_paths.extend(path_batch)
        all_idx = torch.cat((all_idx, idx_batch.cuda()), dim=0)

    out = all_preds, all_labels, all_paths, all_idx

    if not dataframe:
        return out
    else:
        if final_nodes == 1:
            df_out = pd.DataFrame(out[0], columns=['pred'])
        else:
            df_out = pd.DataFrame(
                out[0], columns=[f'pred{i}' for i in range(final_nodes)])
        df_out['y'] = out[1].cpu()
        df_out['fnames'] = out[2]
        df_out['idx'] = out[3].cpu()
        df_out['softmax'] = torch.argmax(
            F.softmax(out[0], dim=1), dim=1).detach().cpu()
        return df_out
