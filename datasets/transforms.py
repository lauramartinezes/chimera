from torchvision import transforms


def set_train_transform(image_size=150, insect_augmentation=True):
    if insect_augmentation:
        return transforms.Compose([
            transforms.Resize(size=(image_size, image_size), antialias=True),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAutocontrast(p=0.7),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
            transforms.RandomRotation(degrees=(-25, 25)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random'),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size=(image_size, image_size), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


def set_test_transform(image_size=150):
    return set_inference_transform(image_size)


def set_feature_extraction_transform():
    return set_inference_transform(224)


def set_inference_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

