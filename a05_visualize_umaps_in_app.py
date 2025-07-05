# Configuration parameters from the YAML-like file
import os
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import torch
import yaml

from PIL import Image
from dash import Dash, dcc, html, Input, Output, no_update


def plot_latent_space_with_tooltips(
    train_features_dict, test_features_dict, labels_train_dict, labels_test_dict, 
    df_train_dict, df_test_dict
):
    from dash import Dash, dcc, html, Input, Output, no_update

    # Initialize the app
    app = Dash(__name__)

    # App layout with dropdown and checklist
    app.layout = html.Div([
        dcc.Dropdown(
            id="insect-class-dropdown",
            options=[{"label": insect, "value": insect} for insect in train_features_dict.keys()],
            value=list(train_features_dict.keys())[0],  # Default selection
        ),
        dcc.Checklist(
            id="train-test-checkbox",
            options=[
                {"label": "Train", "value": "train"},
                {"label": "Test", "value": "test"},
            ],
            value=["train", "test"],  # Default to show both
            inline=True,
        ),
        dcc.Graph(id="graph-latent-space", clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip"),
    ])

    # Global state for curve indices
    curve_indices = {}

    @app.callback(
        Output("graph-latent-space", "figure"),
        Input("insect-class-dropdown", "value"),
        Input("train-test-checkbox", "value")
    )
    def update_graph(selected_class, show_data):
        # Retrieve data for the selected class
        train_features = train_features_dict[selected_class]
        test_features = test_features_dict[selected_class]
        labels_train = labels_train_dict[selected_class]
        labels_test = labels_test_dict[selected_class]

        df_train = df_train_dict[selected_class]
        df_test = df_test_dict[selected_class]

        # Define color palettes
        train_colors = px.colors.qualitative.Set1
        test_colors = px.colors.qualitative.Set2

        train_color_map = {i: train_colors[i] for i in range(len(set(labels_train)))}
        test_color_map = {i: test_colors[i] for i in range(len(set(labels_test)))}

        label_mapping_train = {
            0: f"Normal Sample ({selected_class})",
            1: f"Label Noise",
            2: "Measurement Noise",
        }
        label_mapping_test = {
            0: f"{insect_classes[0]}_test",
            1: f"{insect_classes[1]}_test",
        }

        fig = go.Figure()

        # Reset curve indices for this class
        global curve_indices
        curve_indices = {"train": {}, "test": {}}

        # Add train points if selected
        if "train" in show_data:
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
                    name=label_mapping_train[label],
                    hoverinfo="none",
                )
                fig.add_trace(trace)
                curve_indices["train"][len(fig.data) - 1] = label_indices

        # Add test points if selected
        if "test" in show_data:
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
                    name=label_mapping_test[label],
                    hoverinfo="none",
                )
                fig.add_trace(trace)
                curve_indices["test"][len(fig.data) - 1] = label_indices

        fig.update_layout(
            title="Latent Space UMAP Visualization",
            xaxis=dict(title="UMAP Dimension 1"),
            yaxis=dict(title="UMAP Dimension 2"),
            legend_title="Classes",
            height=700,
            margin=dict(t=50, b=50, l=50, r=50),
        )
        return fig

    @app.callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("graph-latent-space", "hoverData"),
        Input("insect-class-dropdown", "value"),
    )
    def display_hover(hoverData, selected_class):
        if hoverData is None:
            return False, no_update, no_update

        pt = hoverData["points"][0]
        bbox = pt["bbox"]
        curve_number = pt["curveNumber"]
        point_number = pt["pointNumber"]

        # Check if the point belongs to train or test data
        global curve_indices
        if curve_number in curve_indices["train"]:
            global_index = curve_indices["train"][curve_number][point_number]
            df_row = df_train_dict[selected_class].iloc[global_index]
            title = f"Train: {global_index}"
        elif curve_number in curve_indices["test"]:
            global_index = curve_indices["test"][curve_number][point_number]
            df_row = df_test_dict[selected_class].iloc[global_index]
            title = f"Test: {global_index}"
        else:
            return False, no_update, no_update

        # Load and display the image
        img_src = Image.open(df_row["filepath"])
        children = [
            html.Div([
                html.Img(src=img_src, style={"width": "100px", "height": "100px"}),
                html.H4(title),
            ], style={"white-space": "normal"})
        ]
        return True, bbox, children

    app.run_server(debug=False)


if __name__ == '__main__':
    # Load the configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Set manual seed for reproducibility
    torch.manual_seed(config["exp_params"]["manual_seed"])
    random.seed(config["exp_params"]["manual_seed"])
    np.random.seed(config["exp_params"]["manual_seed"])
    data_dir = config["data_params"]["data_dir"]
    swap_suffix = config["data_params"]["swap_suffix"]

    # Load the test DataFrame
    df_test_path = os.path.join(data_dir, f'df_test.csv')
    df_test = pd.read_csv(df_test_path)

    # Define the insect classes
    insect_classes = ['wmv', 'wrl']
    feature_ext_methods = ['cnn', 'adbench']

    # Prepare dictionaries for dropdown-based selection
    train_features_dict = {}
    test_features_dict = {}
    labels_train_dict = {}
    labels_test_dict = {}
    df_train_dict = {}
    df_test_dict = {}

    for feature_ext_method in feature_ext_methods:
        for main_insect_class in insect_classes:
            mislabeled_insect_class = [cls for cls in insect_classes if cls != main_insect_class][0]

            df_train_path = os.path.join(data_dir, f'df_train_{swap_suffix}_{main_insect_class}.csv')
            df_train = pd.read_csv(df_train_path)

            # UMAP file paths
            umap_folder = os.path.join(config["logging_params"]["save_dir"], 'UMAPS')
            umap_train_file_name = os.path.join(umap_folder, f'umap_vect_{main_insect_class}_{feature_ext_method}_train.npy')
            umap_test_file_name = os.path.join(umap_folder, f'umap_vect_{main_insect_class}_{feature_ext_method}_test.npy')
            labels_umap_train_file_name = os.path.join(umap_folder, f'labels_umap_vect_{main_insect_class}_{feature_ext_method}_train.npy')
            labels_umap_test_file_name = os.path.join(umap_folder, f'labels_umap_vect_{main_insect_class}_{feature_ext_method}_test.npy')

            # Check if files exist, then load
            if os.path.exists(umap_train_file_name) and os.path.exists(umap_test_file_name) and \
            os.path.exists(labels_umap_train_file_name) and os.path.exists(labels_umap_test_file_name):
                latents_2d_train = np.load(umap_train_file_name)
                latents_2d_test = np.load(umap_test_file_name)
                labels_noise_train = np.load(labels_umap_train_file_name)
                labels_encoding_test = np.load(labels_umap_test_file_name)

                # Populate dictionaries
                train_features_dict[f'{main_insect_class}_{feature_ext_method}'] = latents_2d_train
                test_features_dict[f'{main_insect_class}_{feature_ext_method}'] = latents_2d_test
                labels_train_dict[f'{main_insect_class}_{feature_ext_method}'] = labels_noise_train
                labels_test_dict[f'{main_insect_class}_{feature_ext_method}'] = labels_encoding_test
                df_train_dict[f'{main_insect_class}_{feature_ext_method}'] = df_train
                df_test_dict[f'{main_insect_class}_{feature_ext_method}'] = df_test
            else:
                print(f"UMAP files not found for {main_insect_class}_{feature_ext_method}. Skipping...")
                continue

    # Call the function with prepared data dictionaries
    plot_latent_space_with_tooltips(
        train_features_dict=train_features_dict,
        test_features_dict=test_features_dict,
        labels_train_dict=labels_train_dict,
        labels_test_dict=labels_test_dict,
        df_train_dict=df_train_dict,
        df_test_dict=df_test_dict
    )
