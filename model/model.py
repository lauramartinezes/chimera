import torch.nn as nn
import torchvision.models as models


def create_model(modelname, weights, topclasses):
    model = model_selector(modelname, weights=weights)
    model_dict = {
        'densenet': modify_densenet,
        'resnet': modify_resnet,
        'efficientnet': modify_efficientnet,
        'vgg': modify_vgg,
        'mobilenet': modify_mobilenet,
    }
    for prefix, modify_fn in model_dict.items():
        if modelname.startswith(prefix):
            model = modify_fn(model, len(topclasses))
            break
    return model


def model_selector(modelname, weights=None):
    model_dict = {
        'densenet121': models.densenet121,
        'densenet169': models.densenet169,
        'mobilenetv3l': models.mobilenet_v3_large,
        'mobilenetv2': models.mobilenet_v2,
        'mobilenetv2quant': models.quantization.mobilenet_v2,
        'vgg16': models.vgg16,
        'vgg19': models.vgg19,
        'efficientnetb0': models.efficientnet_b0,
        'efficientnetb1': models.efficientnet_b1,
        'resnet101': models.resnet101,
        'resnet50': models.resnet50,
    }
    try:
        model_class = model_dict[modelname]
        if modelname == 'mobilenetv2quant':
            return model_class(weights=weights, quantize=True)
        return model_class(weights=weights)
    except KeyError:
        raise ValueError("No model returned")


def modify_densenet(model, num_classes):
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes))
    return model


def modify_resnet(model, num_classes):
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def modify_efficientnet(model, num_classes):
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model


def modify_vgg(model, num_classes):
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    return model


def modify_mobilenet(model, num_classes):
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    return model
