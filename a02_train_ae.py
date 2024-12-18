# Configuration parameters from the YAML-like file
import os
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.utils as vutils

from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import umap
import yaml

from mnist_dataset import CustomBinaryInsectDF
from vq_vae import VQVAE


def train_vae(model, train_loader, device, epochs, optimizer, scheduler, patience, save_path_best):
    best_loss = float('inf')  # Initialize the best loss as infinity
    patience_counter = 0  # Counter for epochs without improvement

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_recons_loss = 0.0
        train_vq_loss = 0.0
        for images, _, _, _, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)

            # Forward pass
            recons, inputs, vq_loss = model(images)
            loss_dict = model.loss_function(recons, inputs, vq_loss)
            loss = loss_dict['loss']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recons_loss += loss_dict['Reconstruction_Loss'].item()
            train_vq_loss += loss_dict['VQ_Loss'].item()

        # Compute average losses for logging
        avg_train_loss = train_loss / len(train_loader)
        avg_recons_loss = train_recons_loss / len(train_loader)
        avg_vq_loss = train_vq_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, "
            f"Reconstruction Loss: {avg_recons_loss:.4f}, VQ Loss: {avg_vq_loss:.4f}")

        # Check for improvement in training loss
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path_best)  # Save the best model
            print(f"New best model saved at {save_path_best} with Train Loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience counter: {patience_counter}/{patience}")
        
        # Step the scheduler
        scheduler.step()
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Early stopping
        if patience_counter >= patience:
            print(f"Stopping early at epoch {epoch+1}. Best Train Loss: {best_loss:.4f}")
            break
    
    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")


def plot_original_vs_reconstructed_images(images, reconstructions, filename=None, dirname=None):
    fig, axes = plt.subplots(2, 8, figsize=(12, 4))
    for i in range(8):
        axes[0, i].imshow(images[i].permute(1, 2, 0).numpy())
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructions[i].permute(1, 2, 0).numpy())
        axes[1, i].axis('off')

    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Reconstructed")
    plt.savefig(os.path.join(dirname, f'original_vs_reconstruction_{filename}.png'), format='png')


def get_train_test_umap(X_train, X_test, n_components=2):
    umap_model = umap.UMAP(n_components=n_components, random_state=42)

    # Fit UMAP on the training data with labels
    X_train_embedding = umap_model.fit_transform(X_train)

    # Transform the test data into the existing UMAP embedding
    X_test_embedding = umap_model.transform(X_test)

    return X_train_embedding, X_test_embedding


import base64
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from PIL import Image
from dash import Dash, dcc, html, Input, Output, no_update

def encode_image_to_base64(image_path):
    """Encode image to base64 string for use in hover."""
    img = Image.open(image_path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def plot_latent_space(train_features, test_features, labels_train, labels_test, df_train, df_test): 
    # Define color palettes
    train_colors = px.colors.qualitative.Set1  # Discrete colors for train data
    test_colors = px.colors.qualitative.Set2  # Discrete colors for test data

    # Create a color map for the test data
    test_color_map = {i: test_colors[i] for i in range(len(set(labels_test)))}  # Map each class to a color for test data
    train_color_map = {i: train_colors[i] for i in range(len(set(labels_train)))}  # Map each class to a color for train data

    label_mapping_train = {0: f"Normal Sample ({main_insect_class})", 1: f"Label Noise ({mislabeled_insect_class})", 2: "Measurement Noise"}
    txt_labels_train = [label_mapping_train[label] for label in labels_train]
    
    label_mapping_test = {0: "wmv_test", 1: "c_test"}
    txt_labels_test = [label_mapping_test[label] for label in labels_test]

    # Encode images for hover using their file paths from the dataframe
    hover_images_train = [encode_image_to_base64(df_train.iloc[i]['filepath']) for i in tqdm(range(len(df_train)), desc="Loading hovering train images")]
    hover_images_test = [encode_image_to_base64(df_test.iloc[i]['filepath']) for i in tqdm(range(len(df_test)), desc="Loading hovering test images")]

    # Create scatterplot for test data
    # Create scatterplot for test data
    test_scatter = go.Scatter(
        x=test_features[:, 0],  # Assuming test_features is 2D
        y=test_features[:, 1],  # Assuming test_features is 2D
        mode='markers',
        marker=dict(
            symbol='circle',
            size=10,
            color=[test_color_map[label] for label in labels_test],  # Use the color map for labels
            line=dict(width=0),
        ),
        hovertemplate=[
            f"Index: {i}<br>{txt_labels_test[i]}<br><img src='data:image/png;base64,{hover_images_test[i]}' width=100 height=100>" for i in range(len(test_features))
        ],  # Show image in hover
        name="Test Data",
        showlegend=False,  # Hide the legend for this trace to create custom legend entries
    )

    # Create scatterplot for train data
    train_scatter = go.Scatter(
        x=train_features[:, 0],  # Assuming test_features is 2D
        y=train_features[:, 1],  # Assuming test_features is 2D
        mode='markers',
        marker=dict(
            symbol='x',
            size=15,
            color=[train_color_map[label] for label in labels_train],  # Use the color map for labels
            line=dict(width=0),
        ),
        hovertemplate=[
            f"Index: {i}<br>{txt_labels_train[i]}<br><img src='data:image/png;base64,{hover_images_train[i]}' width=100 height=100>" for i in range(len(train_features))
        ],  # Show image in hover
        name="Train Data",
        showlegend=False,  # Hide the legend for this trace to create custom legend entries
    )

    # Create dummy scatter traces for legend entries (for each class)
    legend_entries_test = [
        go.Scatter(
            x=[None], y=[None], mode='markers', marker=dict(color=test_colors[i], symbol='circle', size=10),
            name=label_mapping_test[i], showlegend=True
        )
        for i in range(len(label_mapping_test))  # Number of classes for test
    ]

    legend_entries_train = [
        go.Scatter(
            x=[None], y=[None], mode='markers', marker=dict(color=train_colors[i], symbol='x', size=15),
            name=label_mapping_train[i], showlegend=True
        )
        for i in range(len(label_mapping_train))  # Number of classes for train
    ]

    # Combine all traces (data + legend entries)
    fig = go.Figure(data=[test_scatter, train_scatter] + legend_entries_test + legend_entries_train)

    # Add titles and axis labels
    fig.update_layout(
        title="Latent Space UMAP Visualization",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        legend_title="Classes"
    )

    # Show plot
    fig.show()
    # Optionally, save as SVG
    #fig.write_image("latent_space_visualization_with_classes.svg")


def plot_latent_space_with_tooltips(train_features, test_features, labels_train, labels_test, df_train, df_test): 
    # Define color palettes
    train_colors = px.colors.qualitative.Set1
    test_colors = px.colors.qualitative.Set2

    train_color_map = {i: train_colors[i] for i in range(len(set(labels_train)))}
    test_color_map = {i: test_colors[i] for i in range(len(set(labels_test)))}

    # Define label mappings
    label_mapping_train = {0: f"Normal Sample ({main_insect_class})", 1: f"Label Noise ({mislabeled_insect_class})", 2: "Measurement Noise"}
    label_mapping_test = {0: "wmv_test", 1: "c_test"}

    fig = go.Figure()

    # Keep track of curve numbers and their corresponding global indices
    train_curve_indices = {}
    test_curve_indices = {}

    # Add train points with label mappings in the legend
    for label in set(labels_train):
        label_indices = [i for i, l in enumerate(labels_train) if l == label]
        trace = go.Scatter(
            x=train_features[label_indices, 0],
            y=train_features[label_indices, 1],
            mode="markers",
            marker=dict(
                symbol="x",
                size=10,
                color=train_color_map[label],
            ),
            name=label_mapping_train[label],  # Label mapping for train
            hoverinfo="none",
        )
        fig.add_trace(trace)
        train_curve_indices[len(fig.data) - 1] = label_indices  # Map curve number to the indices in the full dataset

    # Add test points with label mappings in the legend
    for label in set(labels_test):
        label_indices = [i for i, l in enumerate(labels_test) if l == label]
        trace = go.Scatter(
            x=test_features[label_indices, 0],
            y=test_features[label_indices, 1],
            mode="markers",
            marker=dict(
                symbol="circle",
                size=10,
                color=test_color_map[label],
            ),
            name=label_mapping_test[label],  # Label mapping for test
            hoverinfo="none",
        )
        fig.add_trace(trace)
        test_curve_indices[len(fig.data) - 1] = label_indices  # Map curve number to the indices in the full dataset

    # Update layout to give more vertical space
    fig.update_layout(
        title="Latent Space UMAP Visualization",
        xaxis=dict(title="UMAP Dimension 1"),
        yaxis=dict(title="UMAP Dimension 2"),
        legend_title="Classes",
        height=700,  # Increase height to give more vertical space
        margin=dict(t=50, b=50, l=50, r=50),  # Adjust margins to reduce unnecessary space around the plot
    )

    app = Dash(__name__)

    app.layout = html.Div([
        dcc.Graph(id="graph-latent-space", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip"),
    ])

    @app.callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("graph-latent-space", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        pt = hoverData["points"][0]
        bbox = pt["bbox"]
        curve_number = pt["curveNumber"]
        point_number = pt["pointNumber"]

        # Check if the point belongs to train or test data
        if curve_number in train_curve_indices:  # Train data
            global_index = train_curve_indices[curve_number][point_number]
            df_row = df_train.iloc[global_index]
            title = f'{label_mapping_train[labels_train[global_index]]}: {global_index}'
        elif curve_number in test_curve_indices:  # Test data
            global_index = test_curve_indices[curve_number][point_number]
            df_row = df_test.iloc[global_index]
            title = f'{label_mapping_test[labels_test[global_index]]}: {global_index}'
        else:
            return False, no_update, no_update

        # Load and display the image
        img_src = Image.open(df_row["filepath"])
        children = [
            html.Div([  # Create a div for the tooltip content
                html.Img(src=img_src, style={"width": "100px", "height": "100px"}),
                html.H4(title),
            ], style={"white-space": "normal"})
        ]

        return True, bbox, children

    app.run_server(debug=True)


def visualize_test_latent_space_wrt_train(train_features, test_features, labels_train, labels_test, filename='', dirname=''):
    umap_folder = os.path.join(dirname, 'UMAPS')
    os.makedirs(umap_folder, exist_ok=True)
    
    # Dictionary to map numbers to text labels
    label_mapping_train = {0: f"Normal Sample ({main_insect_class})", 1: f"Label Noise ({mislabeled_insect_class})", 2: "Measurement Noise"}
    txt_labels_train = [label_mapping_train[label] for label in labels_train]
    
    label_mapping_test = {0: "wmv_test", 1: "c_test"}
    txt_labels_test = [label_mapping_test[label] for label in labels_test]

    # Plot UMAP results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=test_features[:, 0], y=test_features[:, 1], hue=txt_labels_test, palette=sns.color_palette("Set2", len(label_mapping_test)), marker='o', legend='full')
    sns.scatterplot(x=train_features[:, 0], y=train_features[:, 1], hue=txt_labels_train, palette=sns.color_palette("hsv", len(label_mapping_train)), marker='x', s=15, legend='full')
    plt.title("Latent Space UMAP Visualization")
    plt.xlabel("UMAP dimension 1")
    plt.ylabel("UMAP dimension 2")
    plt.savefig(os.path.join(umap_folder, f'umap_plot_{filename}.png'), format='png')
    plt.savefig(os.path.join(umap_folder, f'umap_plot_{filename}.svg'), format='svg')


if __name__ == '__main__':
    # Load the configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Set manual seed for reproducibility
    torch.manual_seed(config["exp_params"]["manual_seed"])
    random.seed(config["exp_params"]["manual_seed"])
    np.random.seed(config["exp_params"]["manual_seed"])

    pin_memory = len(config['trainer_params']['gpus']) != 0

    df_test_path = os.path.join('data', f'df_test.csv')
    df_test = pd.read_csv(df_test_path)

    insect_classes = ['wmv', 'c']
    
    for i in range(len(insect_classes)):
        main_insect_class = insect_classes[i]
        mislabeled_insect_class = insect_classes[1 - i]

        df_train_path = os.path.join('data', f'df_train_ae_{main_insect_class}.csv')
        df_train = pd.read_csv(df_train_path)

        # Prepare Dataset
        transform = transforms.Compose([
            transforms.Resize((config["data_params"]["patch_size"], config["data_params"]["patch_size"])),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        train_dataset_vae = CustomBinaryInsectDF(df_train, transform = transform, seed=config["exp_params"]["manual_seed"])
        test_dataset_vae = CustomBinaryInsectDF(df_test, transform = transform, seed=config["exp_params"]["manual_seed"])

        train_loader = DataLoader(
            train_dataset_vae, 
            batch_size=config["data_params"]["train_batch_size"], 
            shuffle=True,
            num_workers=config["data_params"]["num_workers"],
            pin_memory=pin_memory
        )
        test_loader = DataLoader(
            test_dataset_vae, 
            batch_size=config["data_params"]["train_batch_size"], 
            shuffle=False,
            num_workers=config["data_params"]["num_workers"],
            pin_memory=pin_memory
        )

        print(f"{main_insect_class} dataset correctly loaded")

        # Initialize the Model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = VQVAE(
            in_channels=config["model_params"]["in_channels"],
            embedding_dim=config["model_params"]["embedding_dim"],
            num_embeddings=config["model_params"]["num_embeddings"],
            img_size=config["model_params"]["img_size"],
            beta=config["model_params"]["beta"]
        ).to(device)

        print(f"{main_insect_class} model correctly initialized")

        # Optimizer and Scheduler
        optimizer = optim.Adam(model.parameters(), lr=config["exp_params"]["LR"], weight_decay=config["exp_params"]["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["exp_params"]["scheduler_gamma"])

        # Training Loop
        epochs = config["trainer_params"]["max_epochs"]
        patience = config["trainer_params"]["patience"]
        save_path = os.path.join(config["logging_params"]["save_dir"], f'{config["logging_params"]["name"]}_{main_insect_class}.pth')
        save_path_best = os.path.join(config["logging_params"]["save_dir"], f'{config["logging_params"]["name"]}_{main_insect_class}_best.pth')
        os.makedirs(config["logging_params"]["save_dir"], exist_ok=True)

        if not os.path.exists(save_path_best):
            print(f"{main_insect_class} start training")
            train_vae(model, train_loader, device, epochs, optimizer, scheduler, patience, save_path_best)
        else:
            print(f"{main_insect_class} loading best model")    
        model.load_state_dict(torch.load(save_path_best))

        # Visualization After Training
        model.eval()
        # batch_counter = 0

        # for test_images, labels, _, _, _, _ in tqdm(test_loader):
        #     test_images = test_images.to(device)

        #     with torch.no_grad():
        #         recons, _, _ = model(test_images)

        #     reconstructions_dir = os.path.join(config["logging_params"]["save_dir"] , "Reconstructions")
        #     os.makedirs(reconstructions_dir, exist_ok=True)
        #     vutils.save_image(recons.data,
        #                     os.path.join(reconstructions_dir, f"recons_{main_insect_class}_{batch_counter}.png"),
        #                     normalize=True,
        #                     nrow=12)

        #     # # Denormalize and convert for visualization
        #     test_images = test_images.cpu() * 0.5 + 0.5  # Undo normalization
        #     recons = recons.cpu() * 0.5 + 0.5

        #     # Plot original and reconstructed images
        #     plot_original_vs_reconstructed_images(
        #         test_images, 
        #         recons, 
        #         filename=f'{main_insect_class}_{batch_counter}', 
        #         dirname=os.path.join(config["logging_params"]["save_dir"], "Reconstructions")
        #     )
        #     batch_counter = batch_counter + 1

        # Extract features from encoding latents
        umap_folder = os.path.join(config["logging_params"]["save_dir"], 'UMAPS')
        umap_train_file_name = os.path.join(umap_folder, f'umap_vect_{main_insect_class}_train.npy')
        umap_test_file_name = os.path.join(umap_folder, f'umap_vect_{main_insect_class}_test.npy')
        labels_umap_train_file_name = os.path.join(umap_folder, f'labels_umap_vect_{main_insect_class}_train.npy')
        labels_umap_test_file_name = os.path.join(umap_folder, f'labels_umap_vect_{main_insect_class}_test.npy')


        if os.path.exists(umap_train_file_name) and os.path.exists(umap_test_file_name) and \
            os.path.exists(labels_umap_train_file_name) and os.path.exists(labels_umap_test_file_name): 
            latents_2d_train = np.load(umap_train_file_name)
            latents_2d_test = np.load(umap_test_file_name)
            labels_noise_train = np.load(labels_umap_train_file_name)
            labels_encoding_test = np.load(labels_umap_test_file_name)
        else:            
            (
                raw_features_encoding_test,
                features_encoding_test,
                labels_encoding_test,
                real_labels_encoding_test,
                measurement_noise_encoding_test,
                mislabeled_encoding_test,
            ) = extract_features_from_encoding(model, test_loader, device)

            (
                raw_features_encoding_train,
                features_encoding_train,
                labels_encoding_train,
                real_labels_encoding_train,
                measurement_noise_encoding_train_,
                mislabeled_encoding_train_,
            ) = extract_features_from_encoding(model, train_loader, device)

            measurement_noise_encoding_train = measurement_noise_encoding_train_.astype(int)*2
            mislabeled_encoding_train = mislabeled_encoding_train_.astype(int)
            labels_noise_train = measurement_noise_encoding_train + mislabeled_encoding_train

            # Reduce dimensions to 2D for visualization
            reshaped_raw_features_encoding_test = raw_features_encoding_test.reshape(raw_features_encoding_test.shape[0], -1)
            reshaped_raw_features_encoding_train = raw_features_encoding_train.reshape(raw_features_encoding_train.shape[0], -1)

            filename=f'ae_{main_insect_class}_test'

            latents_2d_train, latents_2d_test = get_train_test_umap(
                reshaped_raw_features_encoding_train, 
                reshaped_raw_features_encoding_test, 
                n_components=2
            )
            
            np.save(umap_train_file_name, latents_2d_train)
            np.save(umap_test_file_name, latents_2d_test)
            np.save(labels_umap_train_file_name, labels_noise_train)
            np.save(labels_umap_test_file_name, labels_encoding_test)

        plot_latent_space_with_tooltips(
            latents_2d_train, 
            latents_2d_test, 
            labels_noise_train, 
            labels_encoding_test,
            df_train, 
            df_test
        )

        plot_latent_space(
            latents_2d_train, 
            latents_2d_test, 
            labels_noise_train, 
            labels_encoding_test,
            df_train, 
            df_test
        )
        
        visualize_test_latent_space_wrt_train(
            latents_2d_train,
            latents_2d_test, 
            labels_noise_train, 
            labels_encoding_test, 
            filename, 
            config["logging_params"]["save_dir"]
        )


    print('')


