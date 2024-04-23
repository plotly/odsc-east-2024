# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc

# Incorporate data
df = px.data.gapminder()

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# App layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                html.Div(
                    "My First App with Data, Graph, and Controls",
                    className="text-primary text-center fs-3",
                )
            ]
        ),
        dbc.Row(
            [
                dbc.RadioItems(
                    options=[
                        {"label": x, "value": x}
                        for x in ["pop", "lifeExp", "gdpPercap"]
                    ],
                    value="lifeExp",
                    inline=True,
                    id="metric-controls",
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dash_table.DataTable(
                            data=df.to_dict("records"),
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
)


# Add controls to build the interaction
@callback(
    Output(component_id="my-graph", component_property="figure"),
    Input(component_id="metric-controls", component_property="value"),
)
def update_graph(col_chosen):
    fig = px.histogram(df, x="continent", y=col_chosen, histfunc="avg")
    return fig


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
