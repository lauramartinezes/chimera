import torchvision.transforms as T

transforms_dict = {
    'common': [T.ToPILImage(), T.Resize(size=(150, 150))],
    'augment': [T.RandomVerticalFlip(p=0.5),
              T.RandomHorizontalFlip(p=0.5),
              T.RandomRotation(degrees=(-15, 15))],
    'to_tensor': [T.ToTensor()]
}


def set_train_transform(augment=False):
    if augment:
        return T.Compose(transforms_dict['common'] + transforms_dict['augment'] + transforms_dict['to_tensor'])
    return T.Compose(transforms_dict['common'] + transforms_dict['to_tensor'])


def set_test_transform():
    return T.Compose(transforms_dict['common'] + transforms_dict['to_tensor'])
