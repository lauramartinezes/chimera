import torch
import torch.nn.functional as F

from models.metrics import compute_accuracy, compute_loss, compute_predictions


def train_epoch(model, train_loader, optimizer, criterion, device, case, epoch, num_epochs):
    model.train()
    epoch_loss = 0
    for batch_idx, (images, labels, _, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
            
        ### FORWARD AND BACK PROP
        logits = model(images)
        probas = F.softmax(logits, dim=1)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        
        loss.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print (f'Case {case} dataset: '+'Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                %(epoch+1, num_epochs, batch_idx, 
                    len(train_loader), loss))  
        epoch_loss += loss.item()
    return epoch_loss / (batch_idx + 1)


def validate_epoch(model, train_loader, val_loader, test_loader, criterion, device, class_weights_tensor, epoch, num_epochs):
    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        val_epoch_loss = compute_loss(model, val_loader, device, criterion=criterion, class_weights_tensor=class_weights_tensor)
        test_epoch_loss = compute_loss(model, test_loader, device, criterion=criterion, class_weights_tensor=class_weights_tensor)
                    
        train_accuracy, train_per_class_accuracy = compute_accuracy(model, train_loader, device=device)
        val_accuracy, val_per_class_accuracy = compute_accuracy(model, val_loader, device=device)
        test_accuracy, test_per_class_accuracy = compute_accuracy(model, test_loader, device=device)


        print('Epoch: %03d/%03d | Train: %.3f%% | Validation: %.3f%% | Test: %.3f%%' % (
            epoch+1, num_epochs, 
            train_accuracy,
            val_accuracy,
            test_accuracy))
        
        print(f'Train per Class Accuracy: ', train_per_class_accuracy)
        print(f'Validation per Class Accuracy: ', val_per_class_accuracy)
        print(f'Test per Class Accuracy: ', test_per_class_accuracy)

        lowest_val_class_accuracy = min(val_per_class_accuracy.values())
    return val_epoch_loss, test_epoch_loss, train_accuracy, val_accuracy, test_accuracy, lowest_val_class_accuracy


def test_model(model, test_loader, device):
    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        (test_accuracy, test_per_class_accuracy) = compute_accuracy(model, test_loader, device)
        print(f'Test accuracy: %.2f%%' % test_accuracy)
        print(f'Test per Class Accuracy: ', test_per_class_accuracy)
    test_predictions, test_actuals, test_probs = compute_predictions(model, test_loader, device)
    return test_accuracy, test_per_class_accuracy, test_predictions, test_actuals, test_probs 
