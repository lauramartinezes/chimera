import numpy as np
import torch

from config.config import INSECT_LABELS_MAP


def test_model(model, loader, dataset):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from tqdm import tqdm

    model.eval()
    correct = 0
    correct_wswl = 0
    total_wswl = 0
    correct_wmv = 0
    total_wmv = 0

    y_pred, y_true = [], []

    feats = ['imgname', 'platename', 'filename', 'plate_idx',
             'location', 'date', 'year', 'xtra', 'width', 'height']
    info = {i: [] for i in feats}

    label_wswl = INSECT_LABELS_MAP['wswl']
    label_wmv = INSECT_LABELS_MAP['wmv']

    for x_batch, y_batch, imgname, platename, filename, plate_idx, location, date, year, xtra, width, height in tqdm(loader, desc='Testing..\t'):
        y_batch = torch.as_tensor(y_batch).type(torch.LongTensor)
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        pred = model(x_batch)
        _, preds = torch.max(pred, 1)
        y_pred.extend(preds.detach().cpu().numpy())
        y_true.extend(y_batch.detach().cpu().numpy())
        correct += (pred.argmax(axis=1) == y_batch).float().sum().item()

        correct_wswl, total_wswl = update_critical_insects_correct_and_total(
            y_batch, pred, label_wswl, correct_wswl, total_wswl)
        correct_wmv, total_wmv = update_critical_insects_correct_and_total(
            y_batch, pred, label_wmv, correct_wmv, total_wmv)

        info['imgname'].extend(imgname)
        info['platename'].extend(platename)
        info['filename'].extend(filename)
        info['plate_idx'].extend(plate_idx)
        info['location'].extend(location)
        info['date'].extend(date)
        info['year'].extend(year)
        info['xtra'].extend(xtra)
        info['width'].extend(width)
        info['height'].extend(height)

    accuracy = correct / len(dataset) * 100.
    accuracy_wswl = calculate_accuracy_critical_insects(
        correct_wswl, total_wswl)
    accuracy_wmv = calculate_accuracy_critical_insects(
        correct_wmv, total_wmv)

    bacc = balanced_accuracy_score(y_pred=y_pred, y_true=y_true)
    cm = confusion_matrix(y_pred=y_pred, y_true=y_true, normalize='true')

    # print(f"Accuracy: {accuracy:.2f}")
    # print(f"Balanced accuracy: {bacc*100.:.2f}")
    # print(f"Confusion matrix: \n{cm}")
    return bacc, cm, y_true, y_pred, info, accuracy_wswl, accuracy_wmv, total_wswl, total_wmv


def update_critical_insects_correct_and_total(y, p, label_insect, correct_insect, total_insect):
    if torch.where(y == label_insect)[0].shape[0] > 0:
        y_insect, p_insect = get_critical_insects_y_and_p(y, p, label_insect)
        correct_insect += (p_insect.argmax(1).eq(y_insect).float()).sum().item()
        total_insect += len(y_insect)
    return correct_insect, total_insect


def get_critical_insects_y_and_p(y, p, label):
    indices_critical_insect = torch.where(y == label)
    return y[indices_critical_insect], p[indices_critical_insect]


def calculate_accuracy_critical_insects(correct, total):
    if total == 0:
        return 0
    else:
        return correct / total * 100
