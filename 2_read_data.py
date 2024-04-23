# Import packages
from dash import Dash, html, dash_table
import plotly.express as px
import pandas as pd

# Incorporate data
df = px.data.gapminder()
# df = pd.read_csv("gapminder.csv")

# Initialize the app
app = Dash(__name__)

# App layout
app.layout = html.Div(
    [
        html.Div(children="My First App with Data"),
        dash_table.DataTable(data=df.to_dict("records"), page_size=10),
    ]
)

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
