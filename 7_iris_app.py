"""=============================================================================
Filename: iris_app.py
Last updated: 2024-04-21

This application allows the user to run k-means clustering on the iris dataset.
============================================================================="""

"""=====================================
Imports
====================================="""
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from sklearn.cluster import KMeans

"""=====================================
Code execution
====================================="""

# Connect to the dataset.
iris_data = px.data.iris()
iris_data.rename(
    columns={
        "sepal_length": "Sepal length (cm)",
        "sepal_width": "Sepal width (cm)",
        "petal_length": "Petal length (cm)",
        "petal_width": "Petal width (cm)",
        "species": "Species",
        "species_id": "Species ID",
    },
    inplace=True,
)

# We don't want species and species ID to be used for the machine learning.
iris_cols = [
    "Sepal length (cm)",
    "Sepal width (cm)",
    "Petal length (cm)",
    "Petal width (cm)",
]

# Initialize the app. Use a Dash Bootstrap theme for styling.
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Iris Clustering"

# Define the app layout using DBC.
app.layout = dbc.Container(
    [
        # Page header
        dbc.Row(
            [html.Div("Iris Clustering", className="text-primary text-center fs-3")]
        ),
        # Control items
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("x-axis"),
                        dcc.Dropdown(
                            options=[{"label": col, "value": col} for col in iris_cols],
                            value="Sepal length (cm)",
                            id="x-dropdown",
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Label("y-axis"),
                        dcc.Dropdown(
                            options=[{"label": col, "value": col} for col in iris_cols],
                            value="Sepal width (cm)",
                            id="y-dropdown",
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Label("Number of clusters"),
                        dbc.Input(id="cluster-input", type="number", value=3),
                    ],
                    width=3,
                ),
                dbc.Col([dbc.Button("Apply", id="apply-button")], width=1, align="end"),
            ]
        ),
        # Scatter plot
        dbc.Row(
            [
                dbc.Col([dcc.Graph(figure={}, id="my-graph")], width=12),
            ]
        ),
        # Table
        dbc.Row(
            [
                dbc.Col(
                    [
                        dash_table.DataTable(
                            id="my-table",
                            page_size=10,
                            style_table={"overflowX": "auto"},
                        )
                    ],
                    width=12,
                ),
            ]
        ),
    ],
    fluid=True,
)  # The fluid option allows the app to fill horizontal space and resize.

"""=====================================
Callback definitions
====================================="""


@app.callback(
    Output("my-graph", "figure"),
    Output("my-table", "data"),
    [
        State("x-dropdown", "value"),
        State("y-dropdown", "value"),
        State("cluster-input", "value"),
        Input("apply-button", "n_clicks"),
    ],
)
def run_clustering(x_var, y_var, num_clusters, n_clicks):
    """
    Runs k-means clustering and updates the graph and table based on the user's
    selections. The user selects the x- and y-variables and chooses the number
    of clusters for the algorithm. When the user clicks the button, the model
    will be run.

    Arguments:
    x_var: The column of the iris dataset to be displayed on the x-axis.
    y_var: The column of the iris dataset to be displayed on the y-axis.
    num_clusters: The number of clusters to be used for k-means clustering.
    n_clicks: Not used in the function. The model is run on button click.
    """
    # Make sure there's at least one cluster.
    num_clusters = max(num_clusters, 1)

    # Make a copy of the iris data for our k-means clustering.
    df = iris_data.copy(deep=True)

    # Perform the k-means clustering and save the values.
    k_means = KMeans(n_clusters=num_clusters)
    k_means.fit(df[[x_var, y_var]].values)
    df["Cluster"] = k_means.labels_.astype(str)

    # Create the scatter plot.
    cluster_ids_sorted = [str(x) for x in list(range(num_clusters))]
    fig = px.scatter(
        df,
        x=x_var,
        y=y_var,
        color="Cluster",
        category_orders={"Cluster": cluster_ids_sorted},
    )

    # Add the centroids to the scatter plot.
    centroid_df = pd.DataFrame(k_means.cluster_centers_, columns=[x_var, y_var])
    centroid_fig = go.Scatter(
        x=centroid_df[x_var],
        y=centroid_df[y_var],
        mode="markers",
        marker={"color": "black", "size": 16, "symbol": "star-triangle-up"},
        name="Centroids",
    )
    fig.add_trace(centroid_fig)

    return fig, df.to_dict("records")


# Run the application.
if __name__ == "__main__":
    app.run_server(debug=True)
