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
pio.templates.default = "plotly_dark"

groups_info = {
    "EdLevel": {
        "Primary/elementary school": "Primary School",
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": "Secondary School",
        "Some college/university study without earning a degree": "Some College/University",
        "Associate degree (A.A., A.S., etc.)": "Associate Degree",
        "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": "Bachelor's Degree",
        "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": "Master's Degree",
        "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": "Professional Degree",
        "Something else": "Other"
    },
    "MainBranch": {
        "I am a developer by profession": "Developer",
        "I am not primarily a developer, but I write code sometimes as part of my work/studies": "Code Sometimes",
        "I used to be a developer by profession, but no longer am": "Ex-Developer",
        "I am learning to code": "Learning to Code",
        "I code primarily as a hobby": "Code as Hobby",
        "None of these": "Other"
    },
    "PurchaseInfluence": {
        "I have little or no influence": "Little or No Influence",
        "I have some influence": "Some Influence",
        "I have a great deal of influence": "Great Influence"
    }
}

def clean_data(groups_info):
    # read in the data from csv to a polars dataframe
    # this file is not on github due to size constraints
    df = pl.read_csv("survey_results_public.csv")
    # shorten/summarize values in ed level, main branch, and purchase influence columns
    for k, v in groups_info.items():
        if type(v) == dict:
            df = df.with_columns(pl.col(k).replace(v))
    # convert converted yearly compensation, years code, and years code pro columns to float
    df = df.with_columns([
        pl.col("ConvertedCompYearly").replace({"NA": None}).cast(pl.Float64).alias("ConvertedCompYearly"),
        pl.col("YearsCode").replace({"NA": None, "Less than 1 year": 0.5, "More than 50 years": 51}).cast(pl.Float64).alias("YearsCode"),
        pl.col("YearsCodePro").replace({"NA": None, "Less than 1 year": 0.5, "More than 50 years": 51}).cast(pl.Float64).alias("YearsCodePro")
    ])
    # create bucket columns for years code and years code pro columns
    # these are not required columns (so include NA column)
    for col in ["YearsCode", "YearsCodePro"]:
        df = df.with_columns(
            pl.when(pl.col(col).is_null()).then(pl.lit("NA")). \
            when(pl.col(col) < 10).then(pl.lit("0-9 Years")). \
            when(pl.col(col) < 20).then(pl.lit("10-19 Years")). \
            when(pl.col(col) < 30).then(pl.lit("20-29 Years")). \
            when(pl.col(col) < 40).then(pl.lit("30-39 Years")). \
            when(pl.col(col) < 50).then(pl.lit("40-49 Years")). \
            otherwise(pl.lit("50+ Years")).alias(col + "Buckets")
        )
    return df

def find_tech_categories(df):
    return sorted(list(set([col.replace("HaveWorkedWith", "").replace("WantToWorkWith", "") for col in df.columns if "HaveWorkedWith" in col or "WantToWorkWith" in col])))

def find_have_want_columns(tech_category):
    return f"{tech_category}HaveWorkedWith", f"{tech_category}WantToWorkWith"

def explode_column(df, column, groups = []):
    # remove rows with missing values in the relevant have or want column
    # this removes respondents that did not fill out the question
    df = df.filter((~pl.col(column).is_null()) & (pl.col(column) != "NA"))
    # keep response id column that indicates the respondent plus (inputted) relevant columns
    return df.with_columns(pl.col(column).str.split(";")).explode(column). \
        select([pl.col(col) for col in ["ResponseId", column] + groups])

def create_have_want_df(df, have_column, want_column, groups = []):
    # expand have and want columns (create a row for each tech listed in the columns)
    have_df, want_df = explode_column(df, have_column, groups = groups), explode_column(df, want_column, groups = groups)
    # full join have and want dataframes 
    have_want_df = have_df.join(other = want_df, left_on = ["ResponseId", have_column] + groups, right_on = ["ResponseId", want_column] + groups, how = "full")
    # coalesce right and left columns to remove duplicate information and missing values
    for col in [col.replace("_right", "") for col in have_want_df.columns if "_right" in col]:
        have_want_df = have_want_df.with_columns(pl.coalesce([col, col + "_right"]).alias(col)).drop(col + "_right")
    return have_want_df

def create_have_want_count_df(tech_category, column, have_want_df, groups = [], clean = False):
    have_want_count_df = have_want_df.filter(~pl.col(column).is_null()).group_by([column] + groups).len()
    if clean:
        return have_want_count_df.rename({column: tech_category}).sort("len", descending = True)
    return have_want_count_df

def create_have_want_prop_df(tech_category, column, have_want_df, groups = [], clean = False, exclusion_prop = None):
    have_want_count_df = create_have_want_count_df(tech_category, column, have_want_df, groups = groups, clean = False)
    # find proportion of total respondents (to the relevant have and want questions) who have or want the tech
    have_want_prop_df = have_want_count_df.with_columns((pl.col("len") / have_want_df["ResponseId"].unique().len()).alias("prop"))
    if exclusion_prop is not None:
        have_want_prop_df = have_want_prop_df.filter(pl.col("prop") >= exclusion_prop)
    if clean:
        return have_want_prop_df.rename({column: tech_category}).drop("len").sort("prop", descending = True)
    return have_want_prop_df

def join_have_want_count_dfs(left, right, left_on, right_on):
    return left.join(other = right, left_on = left_on, right_on = right_on, how = "full")

def clean_prop_df(tech_category, have_column, prop_df, groups = []):
    return prop_df.filter(~pl.col("prop").is_null()).select([pl.col(col) for col in [have_column, "prop"] + groups]).rename({have_column: tech_category}).sort("prop", descending = True)

def create_prop_have_who_want_df(tech_category, have_column, want_column, have_want_df, groups = []):
    # filter where have column is not null (exclude respondents who have not used the tech)
    # only consider the universe of respondents who have used the tech
    have_want_df_filtered = have_want_df.filter(~pl.col(have_column).is_null())
    # find the number of haves and the number of wants who have
    have_count_df = create_have_want_count_df(tech_category, have_column, have_want_df_filtered, groups = groups)
    want_count_df = create_have_want_count_df(tech_category, want_column, have_want_df_filtered, groups = groups)
    # then find the proportion --> number of wants who have / number of haves 
    prop_df = join_have_want_count_dfs(left = have_count_df, right = want_count_df, left_on = [have_column] + groups, right_on = [want_column] + groups). \
        with_columns((pl.col("len_right") / pl.col("len")).alias("prop"))
    return clean_prop_df(tech_category, have_column, prop_df, groups = groups)

def create_prop_want_who_not_have_df(tech_category, have_column, want_column, have_want_df, groups = []):
    # filter where want column is not null (exclude respondents who do not want to use the tech)
    # only consider the universe of respondents who want to use the tech
    have_want_df_filtered = have_want_df.filter(~pl.col(want_column).is_null())
    # find the number of wants and the number of wants who have
    have_count_df = create_have_want_count_df(tech_category, have_column, have_want_df_filtered, groups = groups)
    want_count_df = create_have_want_count_df(tech_category, want_column, have_want_df_filtered, groups = groups)
    # then find the proportion --> 1 - number of wants who have / number of wants = number of wants who do not have / number of wants
    prop_df = join_have_want_count_dfs(left = want_count_df, right = have_count_df, left_on = [want_column] + groups, right_on = [have_column] + groups). \
        with_columns((1 - pl.col("len_right") / pl.col("len")).alias("prop"))
    return clean_prop_df(tech_category, have_column, prop_df, groups = groups)

def create_plot_data(tech_category, metric, groups = [], exclusion_prop = None):
    have_column, want_column = find_have_want_columns(tech_category)
    have_want_df = create_have_want_df(df, have_column, want_column, groups = groups)
    if "Number" in metric:
        y = "len"
        column = have_column if "Have" in metric else want_column
        exclusion_df = create_have_want_prop_df(tech_category, column, have_want_df, exclusion_prop = exclusion_prop, clean = True)
        plot_data = create_have_want_count_df(tech_category, column, have_want_df, groups = groups, clean = True)
    else:
        y = "prop"
        if "Have Who Want" in metric:
            column = have_column
            plot_data = create_prop_have_who_want_df(tech_category, have_column, want_column, have_want_df, groups = groups)
        elif "Want Who Do Not Have" in metric:
            column = want_column
            plot_data = create_prop_want_who_not_have_df(tech_category, have_column, want_column, have_want_df, groups = groups)
        elif "Have" in metric:
            column = have_column
            plot_data = create_have_want_prop_df(tech_category, have_column, have_want_df, groups = groups, clean = True)
        elif "Want" in metric:
            column = want_column
            plot_data = create_have_want_prop_df(tech_category, want_column, have_want_df, groups = groups, clean = True)
        exclusion_df = create_have_want_prop_df(tech_category, column, have_want_df, exclusion_prop = exclusion_prop, clean = True)
    plot_data = plot_data.filter(pl.col(tech_category).is_in(list(exclusion_df[tech_category])))
    if len(groups) > 0:
        return plot_data.pivot(on = tech_category, values = y)
    return plot_data, y
        
def create_plot_col(id, plot, include_metric_select = True):
    card_body = [dcc.Graph(id = f"{id}_{plot}")]
    if include_metric_select:
        card_body.insert(0, dmc.Group([dmc.Select(label = html.B("Analysis Metric"), data = metrics, w = 300, required = True, id = f"{id}_metric")]))
        card_body.insert(1, html.Br())
    return dbc.Col([
        html.Div([
            html.Br(),
            dbc.Card([
                dbc.CardBody(card_body)
            ])
        ], style = {"display": "none"}, id = f"{id}_{plot}_card")
    ], width = 6)

df = clean_data(groups_info)
tech_categories = find_tech_categories(df)
metrics = ["Number Have Worked With", "Proportion Have Worked With", "Number Want To Work With", "Proportion Want To Work With", "Proportion Have Who Want", "Proportion Want Who Do Not Have"]
groups = sorted(["MainBranch", "PurchaseInfluence", "EdLevel", "YearsCodeBuckets", "YearsCodeProBuckets", "Age"])

# create dash app
app = Dash(external_stylesheets = [dbc.themes.CYBORG])

# create simple nav bar
navbar = dbc.Navbar(
    html.Div([
        html.H2("Stack Overflow Developer Survey 2023")
    ], style = {"marginLeft": 10}),
    color = "light"
)

# create app layout
app.layout = html.Div([
    dmc.MantineProvider([
        navbar,
        html.Br(),
        dbc.Card([
            dbc.CardHeader([
                html.B("Technology Analysis")
            ]),
            dbc.CardBody([
                dmc.Group([
                    dmc.Select(label = html.B("Technology Category"), description = "Horizontal axis in bar plots and heat maps", data = tech_categories, w = 300, required = True, id = "tech_category"),
                    dmc.Select(label = html.B("Group"), description = "Vertical axis in heat maps", data = groups, w = 300, required = True, id = "group"),
                    dmc.NumberInput(label = html.B("Exclusion Proportion (0.00-0.25)"), description = "Min proportion of respondents who use tech", min = 0.01, max = 0.25, step = 0.01, id = "exclusion_prop"),
                ]),
                dbc.Row([
                    create_plot_col("left", "bar_plot", include_metric_select = True),
                    create_plot_col("right", "bar_plot", include_metric_select = True),
                ]),
                dbc.Row([
                    create_plot_col("left", "heat_map", include_metric_select = False),
                    create_plot_col("right", "heat_map", include_metric_select = False),
                ])
            ])
        ])
    ], forceColorScheme = "dark")
], style = {"marginLeft": 25, "marginRight": 25, "marginTop": 25, "marginBottom": 25})

# create callbacks to update plots
@app.callback(
    Output(f"left_bar_plot_card", "style", allow_duplicate = True),
    Output(f"right_bar_plot_card", "style", allow_duplicate = True),
    Input("tech_category", "value"),
    prevent_initial_call = True
)
def update_bar_plot_cards_style(tech_category):
    if tech_category is not None:
        return {"display": "block"}, {"display": "block"}
    return {"display": "none"}, {"display": "none"} 

for id in ["left", "right"]:
    @app.callback(
        Output(f"{id}_bar_plot", "figure", allow_duplicate = True),
        Input("tech_category", "value"),
        Input("exclusion_prop", "value"),
        Input(f"{id}_metric", "value"),
        prevent_initial_call = True
    )
    def update_bar_plot(tech_category, exclusion_prop, metric):
        exclusion_prop = None if exclusion_prop == "" else exclusion_prop
        if tech_category is not None and metric is not None:
            plot_data, y = create_plot_data(tech_category, metric, exclusion_prop = exclusion_prop)
            return px.bar(plot_data, x = tech_category, y = y)
        return px.bar()

    @app.callback(
        Output(f"{id}_heat_map", "figure"),
        Output(f"{id}_heat_map_card", "style"),
        Input("tech_category", "value"),
        Input("exclusion_prop", "value"),
        Input(f"{id}_metric", "value"),
        Input("group", "value"),
        prevent_initial_call = True
    )
    def update_heat_map(tech_category, exclusion_prop, metric, group):
        exclusion_prop = None if exclusion_prop == "" else exclusion_prop
        if tech_category is not None and metric is not None and group is not None:
            groups = [group]
            plot_data = create_plot_data(tech_category, metric, groups = groups, exclusion_prop = exclusion_prop)
            text = ".2f" if "Prop" in metric else ".2s"
            return px.imshow(plot_data.drop(groups), y = plot_data[group], x = plot_data.drop(groups).columns, text_auto = text), {"display": "block"}
        return {}, {"display": "none"}

# run dash app
if __name__ == "__main__":
    app.run_server(debug = True)
