# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px

# Incorporate data
df = px.data.gapminder()

# Initialize the app
app = Dash(__name__)

# App layout
app.layout = html.Div(
    [
        html.Div(children="My First App with Data, Graph, and Controls"),
        html.Hr(),
        dcc.Dropdown(
            options=["pop", "lifeExp", "gdpPercap"],
            value="lifeExp",
            id="metric-controls",
        ),
        # dcc.RadioItems(options=["pop", "lifeExp", "gdpPercap"], value="lifeExp", id="metric-controls"),
        html.Br(),
        dash_table.DataTable(data=df.to_dict("records"), page_size=10),
        dcc.Graph(figure={}, id="my-graph"),
    ]
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
