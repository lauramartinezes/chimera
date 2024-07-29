import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

from datasets.transforms import set_train_transform

# Load the EfficientNetV2 model (assumes you have internet access)
model = models.efficientnet_v2_s(pretrained=True)
model.eval()

# Load an example image (ensure internet access to fetch an image)
img = Image.open("/home/u0159868/Documents/data/photobox_vs_fuji/fuji/wmv/2019_beauvechain_w25_B_4183x6278_29541.png")

# Define the image transformations
preprocess = transforms.Compose([
    transforms.Resize(size=(150, 150)),
    transforms.ToTensor(),
])

# Apply the transformations to the image
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

# Dictionary to store the outputs of the layers
layer_outputs = {}

# Define a hook function to save the output of a layer
def get_feature_maps(name):
    def hook(model, input, output):
        layer_outputs[name] = output.detach()
    return hook

# Register hooks to the last layers of the desired blocks
block_last_layers = {
    'block0': 'features.0.2',
    'block2': 'features.2.3',
    'block5': 'features.5.2',
    'block7': 'features.7.2'  # Add block 7's last layer
}
for name, layer in model.named_modules():
    if name in block_last_layers.values():
        layer.register_forward_hook(get_feature_maps(name))

# Perform a forward pass to get the feature maps
with torch.no_grad():
    _ = model(input_batch)

def visualize_feature_maps(feature_maps, ax, title, num_cols=8, cmap='plasma'):
    num_feature_maps = feature_maps.shape[1]
    num_rows = num_feature_maps // num_cols + (num_feature_maps % num_cols > 0)
    ax.set_title(title)
    ax.axis('off')
    for i in range(num_rows):
        for j in range(num_cols):
            if i * num_cols + j < num_feature_maps:
                feature_map = feature_maps[0, i * num_cols + j].cpu().numpy()
                if num_rows > 1:
                    ax_sub = ax.imshow(feature_map, cmap=cmap)
                else:
                    ax.imshow(feature_map, cmap=cmap)
    ax.axis('off')

# Visualize the original image and feature maps of the specified blocks in one plot
fig, axes = plt.subplots(1, 5, figsize=(30, 6))

# Display the original image
axes[0].imshow(img)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Display feature maps of the specified blocks
for idx, (block_name, layer_name) in enumerate(block_last_layers.items()):
    feature_maps = layer_outputs[layer_name]
    visualize_feature_maps(feature_maps, axes[idx + 1], block_name)

plt.show()
