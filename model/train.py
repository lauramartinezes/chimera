import torch
from tqdm import tqdm

from config.config import INSECT_LABELS_MAP
from model.test import update_critical_insects_correct_and_total, calculate_accuracy_critical_insects


def train(model, optimizer, criterion, train_dataset, train_dataloader):
    correct_train = 0
    model.train()
    for x_batch, y_batch, imgname, platename, filename, plate_idx, location, date, year, xtra, width, height in tqdm(train_dataloader, desc='Training..\t'):
        y_batch = torch.as_tensor(y_batch)
        x_batch, y_batch = x_batch.float().cuda(), y_batch.cuda()
        for param in model.parameters():
            param.grad = None
        pred = model(x_batch)

        y_batch = y_batch.type(torch.LongTensor).cuda()
        correct_train += (pred.argmax(axis=1) == y_batch).float().sum().item()
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
    train_accuracy = correct_train / len(train_dataset) * 100.

    return train_accuracy, loss


def validate(model, criterion, valid_dataset, valid_dataloader):
    correct_valid = 0
    correct_wswl = 0
    total_wswl = 0
    correct_wmv = 0
    total_wmv = 0

    label_wswl = INSECT_LABELS_MAP['wswl']
    label_wmv = INSECT_LABELS_MAP['wmv']

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch, imgname, platename, filename, plate_idx, location, date, year, xtra, width, height in tqdm(valid_dataloader, desc='Validating..\t'):
            y_batch = torch.as_tensor(y_batch)
            x_batch, y_batch = x_batch.float().cuda(), y_batch.cuda()
            pred = model(x_batch)

            y_batch = y_batch.type(torch.LongTensor).cuda()
            correct_valid += (pred.argmax(axis=1) ==
                              y_batch).float().sum().item()
            val_loss = criterion(pred, y_batch)

            correct_wswl, total_wswl = update_critical_insects_correct_and_total(
                y_batch, pred, label_wswl, correct_wswl, total_wswl)
            correct_wmv, total_wmv = update_critical_insects_correct_and_total(
                y_batch, pred, label_wmv, correct_wmv, total_wmv)

    valid_accuracy = correct_valid / len(valid_dataset) * 100.
    accuracy_wswl = calculate_accuracy_critical_insects(
        correct_wswl, total_wswl)
    accuracy_wmv = calculate_accuracy_critical_insects(
        correct_wmv, total_wmv)

    return valid_accuracy, val_loss, accuracy_wswl, accuracy_wmv, total_wswl, total_wmv
