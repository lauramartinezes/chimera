import torch
import torch.nn.functional as F
import tqdm


def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    correct_pred_class_0, correct_pred_class_1 = 0, 0
    num_examples_class_0, num_examples_class_1 = 0, 0
    num_examples_measurement_noise_class_0, num_examples_measurement_noise_class_1 = 0, 0
    num_examples_label_noise_class_0, num_examples_label_noise_class_1 = 0, 0
    num_examples_good_class_0, num_examples_good_class_1 = 0, 0
    correct_pred_measurement_noise_class_0, correct_pred_measurement_noise_class_1 = 0, 0
    correct_pred_label_noise_class_0, correct_pred_label_noise_class_1 = 0, 0
    correct_pred_good_class_0, correct_pred_good_class_1 = 0, 0

    for i, (images, labels, _, measurement_noise, label_noise, _) in enumerate(data_loader):
            
        images = images.to(device)
        labels = labels.to(device)
        measurement_noise = measurement_noise.to(device)
        label_noise = label_noise.to(device)

        logits = model(images)
        probas = F.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probas, 1)

        num_examples += labels.size(0)
        num_examples_class_0 += (labels == 0).sum()
        num_examples_class_1 += (labels == 1).sum()
        num_examples_measurement_noise_class_0 += ((measurement_noise == True) & (labels == 0)).sum()
        num_examples_measurement_noise_class_1 += ((measurement_noise == True) & (labels == 1)).sum()
        num_examples_label_noise_class_0 += ((label_noise == True) & (labels == 0)).sum()
        num_examples_label_noise_class_1 += ((label_noise == True) & (labels == 1)).sum()
        num_examples_good_class_0 += ((label_noise == False) & (measurement_noise == False) & (labels == 0)).sum()
        num_examples_good_class_1 += ((label_noise == False) & (measurement_noise == False) & (labels == 1)).sum()

        correct_pred += (predicted_labels == labels).sum()
        correct_pred_class_0 += ((predicted_labels == labels) & (labels == 0)).sum()
        correct_pred_class_1 += ((predicted_labels == labels) & (labels == 1)).sum()
        correct_pred_measurement_noise_class_0 += ((predicted_labels == labels) & (measurement_noise == True) & (labels == 0)).sum()
        correct_pred_measurement_noise_class_1 += ((predicted_labels == labels) & (measurement_noise == True) & (labels == 1)).sum()
        correct_pred_label_noise_class_0 += ((predicted_labels == labels) & (label_noise == True) & (labels == 0)).sum()
        correct_pred_label_noise_class_1 += ((predicted_labels == labels) & (label_noise == True) & (labels == 1)).sum()
        correct_pred_good_class_0 += ((predicted_labels == labels) & (measurement_noise == False) & (label_noise == False) & (labels == 0)).sum()
        correct_pred_good_class_1 += ((predicted_labels == labels) & (measurement_noise == False) & (label_noise == False) & (labels == 1)).sum()
    
    correct_pred_percent = correct_pred.float() / num_examples * 100
    correct_pred_class_0_percent = correct_pred_class_0.float() / num_examples_class_0 * 100
    correct_pred_class_1_percent = correct_pred_class_1.float() / num_examples_class_1 * 100
    correct_pred_measurement_noise_class_0_percent = correct_pred_measurement_noise_class_0.float() / num_examples_measurement_noise_class_0 * 100
    correct_pred_measurement_noise_class_1_percent = correct_pred_measurement_noise_class_1.float() / num_examples_measurement_noise_class_1 * 100
    correct_pred_label_noise_class_0_percent = correct_pred_label_noise_class_0.float() / num_examples_label_noise_class_0 * 100
    correct_pred_label_noise_class_1_percent = correct_pred_label_noise_class_1.float() / num_examples_label_noise_class_1 * 100
    correct_pred_good_class_0_percent = correct_pred_good_class_0.float() / num_examples_good_class_0 * 100
    correct_pred_good_class_1_percent = correct_pred_good_class_1.float() / num_examples_good_class_1 * 100

    return correct_pred_percent, correct_pred_class_0_percent, correct_pred_class_1_percent, correct_pred_measurement_noise_class_0_percent, correct_pred_measurement_noise_class_1_percent, correct_pred_label_noise_class_0_percent, correct_pred_label_noise_class_1_percent, correct_pred_good_class_0_percent, correct_pred_good_class_1_percent

def compute_loss(model, data_loader, device, criterion=None, class_weights_tensor=None):
    epoch_loss = 0
    for batch_idx, (images, labels, _, _, _, _) in enumerate(data_loader):
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

    for images, labels, real_label, measurement_noise, label_noise, outlier  in tqdm.tqdm(data_loader, desc="Computing predictions", total=len(data_loader)):
            
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probas = F.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probas, 1)
        all_predictions.extend(predicted_labels.cpu().numpy())
        all_actuals.extend(labels.cpu().numpy())
        all_probs.extend(probas.cpu().numpy())

    return all_predictions, all_actuals, all_probs
