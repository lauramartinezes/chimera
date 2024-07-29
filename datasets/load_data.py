import numpy as np
from torch.utils.data import DataLoader

from datasets import InsectImgDataset
from datasets.transforms import set_test_transform, set_train_transform


def get_datasets(df_train, df_val, df_test, transform_train=set_train_transform(), transform_test=set_test_transform()):
    return {
        'train_dataset': InsectImgDataset(
            df=df_train.reset_index(drop=True),
            transform=transform_train
        ),
        'valid_dataset': InsectImgDataset(
            df=df_val.reset_index(drop=True),
            transform=transform_test
        ),
        'test_dataset': InsectImgDataset(
            df=df_test.reset_index(drop=True),
            transform=transform_test
        )
    }


def load_data(datasets, batch_size_train, batch_size_val, num_workers):
    dataloaders = {}
    data_params = {
        'train_dataset': {'batch_size': batch_size_train, 'shuffle': True},
        'valid_dataset': {'batch_size': batch_size_val, 'shuffle': False},
        'test_dataset': {'batch_size': batch_size_val, 'shuffle': False},
    }
    for name, dataset in datasets.items():
        params = data_params[name]
        dataloaders[name.split('_')[0] + '_dataloader'] = DataLoader(
            dataset,
            batch_size=params['batch_size'],
            shuffle=params['shuffle'],
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
    return dataloaders


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
