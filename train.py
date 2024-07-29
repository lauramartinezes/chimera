import argparse
import datetime
import json
import pandas as pd
import platform
import timm
import torch.nn as nn
import torch.optim as optim
import wandb

from category_encoders import OrdinalEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR

from config.config import *
from datasets.load_data import *
from datasets.split_dataset import *
from datasets.transforms import set_test_transform, set_train_transform

def get_subset_class_counts(df, subset_size):
    value_counts = df.label.value_counts()
    total_samples = len(df)

    # Creating DataFrame for value counts
    df_counts = pd.DataFrame({
        'label': value_counts.index, 
        'count': value_counts.values, 
        'samples_per_subset_size': np.ceil((value_counts.values / total_samples) * subset_size).astype(int)
        })

    # Calculate rounding error
    rounding_error = abs(subset_size - df_counts['samples_per_subset_size'].sum())

    # Distribute rounding error by removing one sample from each row
    for i in range(min(rounding_error, len(df_counts))):
        df_counts.at[i, 'samples_per_subset_size'] -= 1
    return df_counts

def get_dataset_subset(df, subset_size, random_seed):
    df_counts = get_subset_class_counts(df, subset_size)
    # Create df_train_1000 with samples respecting the proportions
    df_subset = pd.DataFrame()

    for label, count in zip(df_counts['label'], df_counts['samples_per_subset_size']):
        # Sample rows for each label
        sampled_rows = df[df['label'] == label].sample(count, replace=False, random_state=random_seed)
        
        # Append sampled rows to df_train_1000
        df_subset = pd.concat([df_subset, sampled_rows], ignore_index=True)
    return df_subset

import re

def extract_model_size(model_name):
    match = re.search(r'_([a-zA-Z]+)\.', model_name)
    if match:
        return match.group(1).lower()
    else:
        return None
    
def freeze_model_layers(model, last_layers_to_not_freeze):
    # Freeze early layers
    for name, param in model.named_parameters():
        #if 'blocks' in name:  # Assuming 'blocks' are deeper layers in EfficientNetV2
        if name.split('.')[0] == 'blocks' or name.split('.')[0] == 'conv_stem' or name.split('.')[0]=='bn1':
            param.requires_grad = False
    
    # Optionally, unfreeze specified layers
    if last_layers_to_not_freeze != [-1]:
        for name, param in model.named_parameters():
            if any(f'blocks.{layer}' in name for layer in last_layers_to_not_freeze):
                param.requires_grad = True
    
    return model


def get_args():
    parser = argparse.ArgumentParser(description='Pretrained nets and scratch net comparison')
    parser.add_argument('--enable_wandb', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--setting', type=str, default="fuji")
    parser.add_argument('--scenario', type=str, default="scratch", choices=["scratch", "pretrained_imagenet", "pretrained_inaturalist"])
    parser.add_argument('--train_subset_size', type=int, default=None)
    parser.add_argument('--train_subset_seed', type=int, default=156)
    parser.add_argument('--augment', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--last_layers_to_not_freeze', nargs='+', type=int, default=[''], help='List of layer indices to unfreeze')

    args = parser.parse_args()

    if args.augment is False:
        args.augment_name = "non_augmented"
    else:
        args.augment_name = "augmented"

    args.conv_stem = '_conv_stem'
    if args.last_layers_to_not_freeze!=[''] and args.last_layers_to_not_freeze != [-1]:
        args.unfrozen_layers = '_' + '_'.join(map(str, args.last_layers_to_not_freeze))
    elif args.last_layers_to_not_freeze == [-1]:
        args.unfrozen_layers = '_lastlayer'
    elif args.last_layers_to_not_freeze==['']: 
        args.unfrozen_layers = ''
        args.conv_stem = ''

    
    if args.enable_wandb:
        current_time = '{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now())
        if args.train_subset_size is None:
            subset_name = "entire_dataset"
            args.train_subset_seed = 'x'
        else:
            subset_name = args.train_subset_size
        args.trained_model_name = f'{args.setting}_{args.model_name}_{args.scenario}_{subset_name}_seed_{args.train_subset_seed}_{args.augment_name}{args.last_layers_to_not_freeze}{args.conv_stem}_{current_time}'
        args.wandb_run_name = f'run_{args.trained_model_name}'
    else: 
        args.trained_model_name = 'quick_test'
    return args


if __name__ == '__main__':
    system = platform.system()
    args = get_args() 

    batch_size = 32
    batch_size_val = 32
    num_workers = 0

    if args.enable_wandb:
        wandb.init(
            name=args.wandb_run_name,
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY
        )
        wandb.config = {
            "learning_rate": args.lr,
            "epochs": args.num_epochs,
            "batch_size": batch_size
        }

    # Get df_preparation
    df = pd.read_parquet(f"{SAVE_DIR}/df_preparation_{args.setting}.parquet")
    oe = OrdinalEncoder(cols=['label'], mapping=[{'col': 'label', 'mapping': {
                        'bl': 0, 'wswl': 1, 'sp': 2, 't': 3, 'sw': 4, 'k': 5, 'm': 6, 'c': 7, 'v': 8, 'wmv': 9, 'wrl': 10, 'other': 11}}])
    df['txt_label'] = df['label']
    df['label'] = oe.fit_transform(df.label)
    topclasses = df['label'].value_counts().head(12).index.tolist()

    # Get dfs for train, validation, test (here we assume the existance, refer to notebook otherwise)
    df_train = pd.read_parquet(f"{SAVE_DIR}/df_train_{args.setting}.parquet")
    df_val = pd.read_parquet(f"{SAVE_DIR}/df_val_{args.setting}.parquet")
    df_test = pd.read_parquet(f"{SAVE_DIR}/df_test_{args.setting}.parquet")

    # Get training subset
    if args.train_subset_size is not None:
        df_train = get_dataset_subset(df_train, args.train_subset_size, args.train_subset_seed)

    # Get dataloader
    datasets = get_datasets(df_train, df_val, df_test, transform_train=set_train_transform(args.augment), transform_test=set_test_transform())
    train_dataset, valid_dataset, test_dataset = datasets.values()

    dataloaders = load_data(datasets, batch_size, batch_size_val, num_workers)
    train_dataloader, valid_dataloader, test_dataloader = dataloaders.values()

    # Get model
    torch.backends.cudnn.benchmark = True
    
    if args.scenario == "scratch" or args.scenario == 'pretrained_imagenet':
        if args.scenario == "scratch":
            model = timm.create_model(args.model_name, pretrained=False, num_classes=len(topclasses))
        elif args.scenario == 'pretrained_imagenet':
            model = timm.create_model(args.model_name, pretrained=True, num_classes=len(topclasses))
            if args.last_layers_to_not_freeze:
                model = freeze_model_layers(model, args.last_layers_to_not_freeze)
        # # Check if layers are actually frozen
        # for name, param in model.named_parameters():
        #     if not param.requires_grad:
        #         print(f"Layer '{name}' is frozen.")
        #     else:
        #         print(f"Layer '{name}' is trainable.") 

        # Choosing whether to train on a gpu
        train_on_gpu = torch.cuda.is_available()
        print(f'Train on gpu: {train_on_gpu}')  # Number of gpus
        model = model.to('cuda', dtype=torch.float)
        

    elif args.scenario == 'pretrained_inaturalist':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Let's load the model
        model_directory = os.path.normpath(r'/home/u0159868/Documents/data/inaturalist_models')        
        model_size = extract_model_size(args.model_name)
        inaturalist_model_name = f'EfficientNetV2_{model_size}_tf_efficientnetv2_{model_size}.in21k_ft_in1k_32_full_CrossEntropyLoss_sgd_cyclic_latest'
        model_path = os.path.join(model_directory, f'{inaturalist_model_name}_model.pth')
        lbl_mapping_path = os.path.join(model_directory, f'{inaturalist_model_name}_label_mapping.json')
        with open(lbl_mapping_path, 'r') as f:
            lbl_mapping = json.load(f)
        num_classes_inaturalist = len(lbl_mapping)
        # Let's load the model
        model = timm.create_model(args.model_name, num_classes=num_classes_inaturalist) 
        model.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.2, inplace=False), model.classifier)

        z = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(z['model'])
        
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(topclasses))
        if args.last_layers_to_not_freeze:
            model = freeze_model_layers(model, args.last_layers_to_not_freeze)
        model = model.to('cuda', dtype=torch.float)

    class_sample_count = np.unique(df_train.label, return_counts=True)[1]
    weight = 1. / class_sample_count
    criterion = nn.CrossEntropyLoss(
        label_smoothing=.15, weight=torch.Tensor(weight).cuda())

    if not args.last_layers_to_not_freeze or args.scenario == "scratch":
        optimizer = optim.AdamW(model.parameters(), lr=.003, weight_decay=0.05)
    else: 
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.003, weight_decay=0.05)
        # Check what params are optimized
        # for i, param_group in enumerate(optimizer.param_groups):
        #     print(f"Group {i}:")
        #     trainable_indices = [str(idx) for idx, param in enumerate(param_group['params']) if param.requires_grad]
        #     frozen_indices = [str(idx) for idx, param in enumerate(param_group['params']) if not param.requires_grad]
        #     print(f"  Trainable parameter indices: {', '.join(trainable_indices)}")
        #     print(f"  Frozen parameter indices: {', '.join(frozen_indices)}") 
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=args.lr, max_lr=0.03, cycle_momentum=False, mode="triangular2")
    



    # Training 
    results = {"loss": [], "val_loss": [],
            "train_accuracy": [], "valid_accuracy": []}
    best_valacc = 0
    # Model training
    for epoch in range(args.num_epochs):
        # Going through the training set
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
        
        torch.cuda.empty_cache()

        # Going through the validation set
        correct_valid = 0
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
        valid_accuracy = correct_valid / len(valid_dataset) * 100.

        if args.enable_wandb:
            wandb.log({
                'train_loss': loss.item(),
                'train_accuracy': train_accuracy,
                'val_loss': val_loss.item(),
                'valid_accuracy': valid_accuracy,
            })

        scheduler.step()

        # Printing results
        print(f"Epoch {epoch}: train_acc: {train_accuracy:.1f}% loss: {loss:.7f},  val_loss: {val_loss:.7f} val_acc: {valid_accuracy:.1f}%")

        is_best = valid_accuracy > best_valacc
        if is_best:
            print(
                f"Validation accuracy improved from {best_valacc:.2f} to {valid_accuracy:.2f}. Saving model..")
        best_valacc = max(valid_accuracy, best_valacc)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_valacc': best_valacc,
            'loss': results['loss'].append(loss.detach().cpu()),
            'val_loss': results['val_loss'].append(val_loss.detach().cpu()),
            'train_accuracy': results['train_accuracy'].append(train_accuracy),
            'valid_accuracy': results['valid_accuracy'].append(valid_accuracy),
            'optimizer': optimizer.state_dict(),
        }, is_best, f"{args.setting}_{args.model_name}_{args.scenario}_{args.train_subset_size}_seed_{args.train_subset_seed}_{args.augment_name}{args.unfrozen_layers}{args.conv_stem}")