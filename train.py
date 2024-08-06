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


def get_args():
    parser = argparse.ArgumentParser(description='Pretrained nets and scratch net comparison')
    parser.add_argument('--enable_wandb', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--setting', type=str, default="fuji")
    parser.add_argument('--scenario', type=str, default="pretrained_imagenet", choices=["scratch", "pretrained_imagenet", "pretrained_inaturalist"])
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


# Note: For now we are not using the pretrained weights of the previous iteration, to be discussed in next progress meeting
def train(df_train, df_val, df_test, iteration, save_folder):
    args = get_args() 

    args.wandb_run_name = args.wandb_run_name + f'_{iteration}'

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

    # Get dataloader
    datasets = get_datasets(df_train, df_val, df_test, transform_train=set_train_transform(args.augment), transform_test=set_test_transform())
    train_dataset, valid_dataset, test_dataset = datasets.values()

    dataloaders = load_data(datasets, batch_size, batch_size_val, num_workers)
    train_dataloader, valid_dataloader, test_dataloader = dataloaders.values()

    # Get model
    torch.backends.cudnn.benchmark = True
    model = timm.create_model(args.model_name, pretrained=True, num_classes=len(topclasses))

    # Choosing whether to train on a gpu
    train_on_gpu = torch.cuda.is_available()
    print(f'Train on gpu: {train_on_gpu}')  # Number of gpus
    model = model.to('cuda', dtype=torch.float)

    class_sample_count = np.unique(df_train.label, return_counts=True)[1]
    weight = 1. / class_sample_count
    criterion = nn.CrossEntropyLoss(
        label_smoothing=.15, weight=torch.Tensor(weight).cuda())

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.003, weight_decay=0.05)
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

        model_name = f"{args.setting}_{args.model_name}_{args.scenario}_{args.train_subset_size}_seed_{args.train_subset_seed}_{args.augment_name}{args.unfrozen_layers}{args.conv_stem}_{iteration}"
        if is_best:
            print(
                f"Validation accuracy improved from {best_valacc:.2f} to {valid_accuracy:.2f}. Saving model..")
        best_valacc = max(valid_accuracy, best_valacc)
        save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_valacc': best_valacc,
                'loss': results['loss'].append(loss.detach().cpu()),
                'val_loss': results['val_loss'].append(val_loss.detach().cpu()),
                'train_accuracy': results['train_accuracy'].append(train_accuracy),
                'valid_accuracy': results['valid_accuracy'].append(valid_accuracy),
                'optimizer': optimizer.state_dict(),
            }, 
            is_best, 
            f"{args.setting}_{args.model_name}_{args.scenario}_{args.train_subset_size}_seed_{args.train_subset_seed}_{args.augment_name}{args.unfrozen_layers}{args.conv_stem}_{iteration}",
            folder=save_folder
        )

        return os.path.join(save_folder, f'{model_name}_best.pth.tar')