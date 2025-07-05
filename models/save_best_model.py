import torch


def save_best_model(model, save_path_best, best_val_accuracy, val_accuracy, lowest_val_class_accuracy, epochs_no_improve):
    if best_val_accuracy < val_accuracy:
        best_val_accuracy = val_accuracy
        best_lowest_val_class_accuracy = lowest_val_class_accuracy
        torch.save(model.state_dict(), save_path_best)
        print(f"New best model saved at {save_path_best} with Validation Accuracy: {best_val_accuracy:.3f}% and lowest class accuracy: {best_lowest_val_class_accuracy:.3f}%")
        epochs_no_improve = 0  # reset patience counter
    else:
        epochs_no_improve += 1
        print(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s)")
    return best_val_accuracy, epochs_no_improve