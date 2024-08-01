# import libraries
import polars as pl
import us
from dash import Dash, dcc, html, Input, Output, _dash_renderer, no_update, ctx
from dash._callback import PreventUpdate
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.express as px
import plotly.io as pio
# set react version for dmc components
_dash_renderer._set_react_version("18.2.0")
# set plotly theme
pio.templates.default = "plotly_white"

def find_state_code(state_name):
    # map state name to state code
    state = us.states.lookup(state_name)
    # check that state is not None and that state is not a US territory
    if state is not None and not state.is_territory:
        return state.abbr
    return ""

def clean_data():
    # read in the csv data
    df = pl.read_csv("rural-investments.csv", ignore_errors = True)

    # convert "County FIPS" to string
    # convert "Investment Dollars" to float
    # create "State Code" column
    df = df.with_columns([
        pl.col("County FIPS").str.replace("'", "").str.zfill(5),
        pl.col("Investment Dollars").str.replace_all(",", "").cast(pl.Float64),
        pl.col("State Name").map_elements(find_state_code, return_dtype = pl.Utf8).alias("State Code")
    ])

    # remove data for US territories (which do not show up in the US maps)
    df = df.filter(pl.col("State Code") != "")

    return df

def group_and_calc_data(df, group_by):
    # create sum, count, and average columns
    return df.group_by(group_by).agg([
        pl.col("Investment Dollars").sum().alias("Sum Investment Dollars"), 
        pl.col("Number of Investments").sum(),
        (pl.col("Investment Dollars").sum() / pl.col("Number of Investments").sum()).alias("Average Investment Dollars"),
    ])

def create_plot_data(df, group_by, state = None):
    # filter for state if state is not None
    if state is not None:
        data = df.filter(pl.col("State Code") == state)
    else:
        data = df
    # create sum, count, and average columns
    # add proportion socially vulnerable columns for sum and count
    data = group_and_calc_data(data, group_by).join(
        other = group_and_calc_data(data.filter(pl.col("Svi Status") == "Socially Vulnerable"), group_by),
        on = group_by,
        how = "left"
    ).with_columns([
        (pl.col("Sum Investment Dollars_right") / pl.col("Sum Investment Dollars")).fill_null(0).alias("Prop Investment Dollars SVI"),
        (pl.col("Number of Investments_right") / pl.col("Number of Investments")).fill_null(0).alias("Prop Number of Investments SVI"),
    ]).drop(["Sum Investment Dollars_right", "Number of Investments_right", "Average Investment Dollars_right"])
    return data

def create_map(plot_data, metric, state = None):
    # create plotly express choropleth map at the state or county level
    if state is None:
        # state level
        params = dict(
            locations = "State Code", 
            locationmode = "USA-states",
            hover_data = ["State Name"],
            title = f"{metric} by State"
        )
    else:
        # county level
        params = dict(
            geojson = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json", 
            locations = "County FIPS",
            hover_data = ["County"],
            title = f"{metric} by County (State = {state})"
        )
    fig = px.choropleth(
        plot_data,
        color = metric,
        scope = "usa",
        color_continuous_scale = "Turbo",
        **params
    )
    # add black country and state borders
    fig.update_geos(
        visible = False,
        showcountries = True, countrycolor = "Black",
        showsubunits = True, subunitcolor = "Black"
    )
    fig.update_traces(marker_line_color = "Black")
    # remove colorbar title and set thickness
    fig.update_layout(coloraxis_colorbar = dict(title = "", thicknessmode = "pixels", thickness = 25))
    # zoom to state if state map
    if state is not None and state != "AK": # fitbounds = "locations" behaves strangely for Alaska
        fig.update_geos(fitbounds = "locations")
    return fig

def create_bar_chart(plot_data, metric, variable, state = None):
    # create plotly express bar chart
    plot_data = plot_data.sort(metric, descending = False)
    # add text to bars
    text_auto = ".2s" if "Prop" not in metric else ".2f"
    # create title
    title = f"{metric} by {variable}"
    if state is not None:
        title += f" (State = {state})"
    fig = px.bar(plot_data, x = metric, y = variable, text_auto = text_auto, title = title)
    # remove legend
    fig.update_layout(showlegend = False)
    # change color of bars to black
    fig.update_traces(marker_color = "Black", marker_line_color = "Black")
    return fig

# create selection row
def create_selection_row(card_header, options, default_value, id):
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.B(card_header)
                ]),
                dbc.CardBody([
                    dmc.RadioGroup(
                        children = dmc.Group([dmc.Radio(label = x, value = x) for x in options]),
                        value = default_value,
                        id = id
                    )
                ])
            ])
        ])
    ])

# create plots row (2 plots per row)
def create_plots_row(left_id, right_id):
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id = left_id)
                ])
            ])
        ], width = 12, id = left_id + "_col"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id = right_id)
                ])
            ])
        ], style = {"display": "none"}, id = right_id + "_col")
    ])

# create data
df = clean_data()

# create dash app
app = Dash(external_stylesheets = [dbc.themes.CERULEAN])

# create simple nav bar
navbar = dbc.Navbar(
    html.Div([
        html.H2("US Rural Investments 2024")
    ], style = {"marginLeft": 10}),
    color = "dark"
)

metrics = [
    "Sum Investment Dollars", 
    "Number of Investments", 
    "Average Investment Dollars", 
    "Prop Investment Dollars SVI",
    "Prop Number of Investments SVI"
]

variables = [
    "Program Area", 
    "Svi Status",
    "Investment Type"
]

# create app layout
app.layout = html.Div([
    dmc.MantineProvider([
        navbar,
        html.Br(),
        create_selection_row("Metric Selection", metrics, "Sum Investment Dollars", "metric"),
        html.Br(),
        create_plots_row("state_map", "county_map"),
        html.Br(),
        create_selection_row("Bar Chart Variable Selection", variables, "Program Area", "variable"),
        html.Br(),
        create_plots_row("overall_bar_chart", "state_bar_chart")
    ])
], style = {"marginLeft": 25, "marginRight": 25, "marginTop": 25, "marginBottom": 25})

# create app callbacks
@app.callback(
    Output("state_map", "figure"),
    Output("overall_bar_chart", "figure"),
    Input("metric", "value"),
    Input("variable", "value")
)
def update_overall_plots(metric, variable):
    if ctx.triggered_id == "variable":
        state_map = no_update
    else:
        state_map_plot_data = create_plot_data(df, group_by = ["State Name", "State Code"])
        state_map = create_map(state_map_plot_data, metric = metric)
    overall_bar_chart_data = create_plot_data(df, group_by = variable)
    overall_bar_chart = create_bar_chart(overall_bar_chart_data, metric = metric, variable = variable)
    return state_map, overall_bar_chart

@app.callback(
    output = {
        "county_map": {
            "figure": Output("county_map", "figure"),
            "left_col_width": Output("state_map_col", "width"),
            "right_col_width":Output("county_map_col", "width"),
            "right_col_style": Output("county_map_col", "style")
        },
        "state_bar_chart": {
            "figure": Output("state_bar_chart", "figure"),
            "left_col_width": Output("overall_bar_chart_col", "width"),
            "right_col_width": Output("state_bar_chart_col", "width"),
            "right_col_style": Output("state_bar_chart_col", "style")
        }
    },
    inputs = {
        "click_data": Input("state_map", "clickData"),
        "metric": Input("metric", "value"),
        "variable": Input("variable", "value"),
    },
    prevent_initial_call = True
)
def update_state_plots(click_data, metric, variable):
    if click_data is not None:
        state = click_data["points"][0]["location"]
        if ctx.triggered_id == "variable":
            county_map = no_update
        else:
            county_map_plot_data = create_plot_data(df, group_by = ["State Name", "State Code", "County", "County FIPS"], state = state)
            county_map = create_map(county_map_plot_data, metric = metric, state = state)
        state_map_plot_data = create_plot_data(df, group_by = variable, state = state)
        state_bar_chart = create_bar_chart(state_map_plot_data, metric = metric, variable = variable, state = state)
        col_info = dict(left_col_width = 6, right_col_width = 6, right_col_style = {"display": "block"})
        return {
            "county_map": {
                "figure": county_map,
                **col_info
            },
            "state_bar_chart": {
                "figure": state_bar_chart,
                **col_info
            }
        }
    raise PreventUpdate

# run dash app
if __name__ == "__main__":
    app.run_server(debug = True)
