import os
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns


from matplotlib import pyplot as plt
from PIL import Image


def plot_test_latent_space_wrt_train(main_insect_class, mislabeled_insect_class, train_features, test_features, labels_train, labels_test, insect_classes, filename='', dirname=''):
    umap_folder = os.path.join(dirname, 'UMAPS')
    os.makedirs(umap_folder, exist_ok=True)
    
    # Dictionary to map numbers to text labels
    label_mapping_train = {0: f"Normal Sample ({main_insect_class})", 1: f"Label Noise ({mislabeled_insect_class})", 2: "Measurement Noise"}
    txt_labels_train = [label_mapping_train[label] for label in labels_train]
    
    label_mapping_test = {0: f"{insect_classes[0]}_test", 1: f"{insect_classes[1]}_test"}
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


def plot_train_latent_space(train_features, labels_train, filename='', dirname=''):
    umap_folder = os.path.join(dirname, 'UMAPS')
    os.makedirs(umap_folder, exist_ok=True)
    
    # Dictionary to map numbers to text labels
    label_mapping = {0: "Normal Sample", 1: "Label Noise", 2: "Measurement Noise"}
    txt_noise_labels = [label_mapping[label] for label in labels_train]

    # Plot UMAP results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=train_features[:, 0], y=train_features[:, 1], hue=txt_noise_labels, palette=sns.color_palette("hsv", 3), legend='full')
    plt.title("Latent Space UMAP Visualization")
    plt.xlabel("UMAP dimension 1")
    plt.ylabel("UMAP dimension 2")
    plt.savefig(os.path.join(umap_folder, f'umap_plot_{filename}.png'), format='png')
    plt.savefig(os.path.join(umap_folder, f'umap_plot_{filename}.svg'), format='svg')


def plot_latent_space_with_tooltips(
    train_features_dict, test_features_dict, labels_train_dict, labels_test_dict, 
    df_train_dict, df_test_dict, insect_classes
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


