import torch
import torch.nn.functional as F
import tqdm

from collections import defaultdict


def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    correct_pred_per_class = defaultdict(int)
    num_examples_per_class = defaultdict(int)

    for images, labels, measurement_noise, label_noise  in tqdm.tqdm(data_loader, desc="Computing accuracy", total=len(data_loader)):        
        images = images.to(device)
        labels = labels.to(device)
        measurement_noise = measurement_noise.to(device)
        label_noise = label_noise.to(device)

        logits = model(images)
        probas = F.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probas, 1)

        correct_pred += (predicted_labels == labels).sum().item()
        num_examples += labels.size(0)
        
        for class_idx in labels.unique():
            class_idx = class_idx.item()
            class_mask = (labels == class_idx)

            num_examples_per_class[class_idx] += class_mask.sum().item()
            correct_pred_per_class[class_idx] += ((predicted_labels == labels) & class_mask).sum().item()
    
    correct_pred_percent = correct_pred / num_examples * 100
    
    # Compute per-class accuracy
    per_class_accuracy = {}
    for class_idx in num_examples_per_class:
        per_class_accuracy[class_idx] = 100.0 * correct_pred_per_class[class_idx] / num_examples_per_class[class_idx]

    return correct_pred_percent, per_class_accuracy

def compute_loss(model, data_loader, device, criterion=None, class_weights_tensor=None):
    epoch_loss = 0
    for batch_idx, (images, labels, _, _) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        logits = model(images)
        if criterion is not None:
            loss = criterion(logits, labels)
        else:
            loss = F.cross_entropy(logits, labels, weight=class_weights_tensor)  # Compute validation loss
        epoch_loss += loss.item()

    epoch_loss /= (batch_idx + 1)  # Compute average validation loss
    return epoch_loss


def compute_predictions(model, data_loader, device):
    all_predictions = []
    all_actuals = []
    all_probs = []

    for images, labels, _, _  in tqdm.tqdm(data_loader, desc="Computing predictions", total=len(data_loader)):  
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probas = F.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probas, 1)
        all_predictions.extend(predicted_labels.detach().cpu().numpy())
        all_actuals.extend(labels.detach().cpu().numpy())
        all_probs.extend(probas.detach().cpu().numpy())

    return all_predictions, all_actuals, all_probs
