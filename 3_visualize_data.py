# Import packages
from dash import Dash, html, dash_table, dcc
import pandas as pd
import plotly.express as px

# Incorporate data
df = px.data.gapminder()

# Initialize the app
app = Dash(__name__)

# App layout
app.layout = html.Div(
    [
        html.Div(children="My First App with Data and a Graph"),
        dash_table.DataTable(data=df.to_dict("records"), page_size=10),
        dcc.Graph(figure=px.histogram(df, x="continent", y="lifeExp", histfunc="avg")),
    ]
)

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
