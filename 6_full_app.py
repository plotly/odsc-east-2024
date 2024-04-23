"""=============================================================================
Filename: full_app.py
Last updated: 2024-04-21

This application visualizes data from the Gapminder dataset. Components on the
page include a graph, chart, and table. Users can update the content displayed
on the page using interactive UI components, which trigger callback functions.
============================================================================="""

"""=====================================
Imports
====================================="""
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import plotly.express as px
import dash_bootstrap_components as dbc

"""=====================================
Code execution
====================================="""

# Connect to the dataset.
df = px.data.gapminder()
years = df["year"].unique().tolist()
continents = df["continent"].unique().tolist()

# Initialize the app. Use a Dash Bootstrap theme for styling.
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "World Population Data"

# Define the app layout using DBC.
app.layout = dbc.Container(
    [
        # Page header
        dbc.Row(
            [
                html.Div(
                    "World Population Data", className="text-primary text-center fs-3"
                )
            ]
        ),
        # Control items
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Metric"),
                        dcc.Dropdown(
                            options=[
                                {"label": x, "value": x}
                                for x in ["pop", "lifeExp", "gdpPercap"]
                            ],
                            value="lifeExp",
                            id="metric-controls",
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Label("Year"),
                        dcc.Dropdown(
                            options=[year for year in years] + ["All"],
                            value="All",
                            id="year-dropdown",
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Label("Continents"),
                        dcc.Dropdown(
                            options=[continent for continent in continents],
                            value=[continent for continent in continents],
                            multi=True,
                            id="continent-dropdown",
                        ),
                    ],
                    width=5,
                ),
                dbc.Col([dbc.Button("Apply", id="apply-button")], width=1, align="end"),
            ]
        ),
        # Map
        dbc.Row([dcc.Graph(figure={}, id="my-map")]),
        # Graph and table
        dbc.Row(
            [
                dbc.Col(
                    [
                        dash_table.DataTable(
                            data=df.to_dict("records"),
                            id="my-table",
                            page_size=10,
                            style_table={"overflowX": "auto"},
                        )
                    ],
                    width=6,
                ),
                dbc.Col([dcc.Graph(figure={}, id="my-graph")], width=6),
            ]
        ),
    ],
    fluid=True,
)  # The fluid option allows the app to fill horizontal space and resize.

"""=====================================
Callback definitions
====================================="""


# Update the outputs on the page. By uncommenting the commented-out Inputs and
# function definition and commenting out the Input, States, and existing
# function definition, the app will be changed such that the callback will
# update automatically when any of the dropdown values change, instead of only
# when the button is pressed.
@callback(
    Output(component_id="my-graph", component_property="figure"),
    Output(component_id="my-map", component_property="figure"),
    Output(component_id="my-table", component_property="data"),
    #     Input(component_id="metric-controls", component_property="value"),
    #     Input(component_id="year-dropdown", component_property="value"),
    #     Input(component_id="continent-dropdown", component_property="value"),
    # )
    # def update_outputs(col_chosen, year_chosen, continents_chosen):
    Input(component_id="apply-button", component_property="n_clicks"),
    State(component_id="metric-controls", component_property="value"),
    State(component_id="year-dropdown", component_property="value"),
    State(component_id="continent-dropdown", component_property="value"),
)
def update_outputs(n_clicks, col_chosen, year_chosen, continents_chosen):
    """
    Updates the graph, map, and table based on the user's selections.

    Arguments:
    n_clicks: Not used. The callback triggers on button click.
    col_chosen: The column the user has selected to display.
    year_chosen: The year or years from which to display data.
    continents_chosen: The continents from which to display data.
    """

    if year_chosen == "All":
        filtered_df = df[df["continent"].isin(continents_chosen)]
    else:
        filtered_df = df[
            (df["year"] == year_chosen) & (df["continent"].isin(continents_chosen))
        ]
    grouped_df = (
        filtered_df.groupby(["country", "continent", "iso_alpha"])[
            ["pop", "lifeExp", "gdpPercap"]
        ]
        .mean()
        .reset_index()
    )

    hist_fig = px.histogram(grouped_df, x="continent", y=col_chosen, histfunc="avg")
    choropleth_map = px.choropleth(
        grouped_df,
        locations="iso_alpha",
        color_continuous_scale="Viridis",
        color=col_chosen,
        hover_name="country",
    )
    table_data = filtered_df.to_dict("records")
    return hist_fig, choropleth_map, table_data


# Run the application.
if __name__ == "__main__":
    app.run_server(debug=True)
