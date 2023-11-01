from . import html, dcc, dash_table, go

from utils.appUtils import Datasets


def get_layout():

    return dcc.Tab(label='Dataset Characteristics', value="dataset", children=[

        html.Div(
            style={
                "display": "flex",
                "align-items": "center",
                'flex-direction': 'column',
                "justify-content": "center",
                "height": "30vh",  # Adjust the height as needed
            },
            children=[
                html.Button('Initialize Characteristics Tab', id='initialize_characteristics_tab', n_clicks=0, style={
                    'background-color': '#3DAFA8',
                    'color': 'white',
                    'border': 'none',
                    'padding': '10px 20px',
                    'text-align': 'center',
                    'text-decoration': 'none',
                    'display': 'inline-block',
                    'font-size': '16px',
                    'margin': '4px 2px',
                    'cursor': 'pointer',
                    'border-radius': '4px'
                }),
                html.Div(id='initialize_dataset')
            ]),


        html.H3("Statistical Summary", style={'text-align': 'center'}),

        html.Div(
            style={"width": "30%", 'display': 'inline-block', },
            children=[
                dcc.Dropdown(
                    id='statistical_columns',
                    multi=True,
                    placeholder="Select column...",
                ),
                html.Button('Generate Statistical Summary', id='generate_summary', n_clicks=0, style={
                    'background-color': '#3DAFA8',
                    'color': 'white',
                    'border': 'none',
                    'padding': '10px 20px',
                    'text-align': 'center',
                    'text-decoration': 'none',
                    'display': 'inline-block',
                    'font-size': '16px',
                    'margin': '4px 2px',
                    'cursor': 'pointer',
                    'border-radius': '4px'
                }),
                dcc.Checklist(
                    id='include_ignored_stats',
                    options=[
                        {'label': 'Include IGNORED Tokens', 'value': 'checked'}
                    ],
                    value=[]
                ),
            ]),


        dash_table.DataTable(
            id='describe_table',
            columns=[],
            style_header={'text-align': 'center', 'background-color': '#555555',
                          'color': 'white'},
            data=[],
            editable=True,
            filter_action="native",
            sort_action="native",
            sort_mode='multi',
            column_selectable="single",
            row_selectable='multi',
            row_deletable=True,
            selected_rows=[],
            page_action='native',
            page_current=0,
            page_size=10,
        ),

        html.H3("Distributions", style={'text-align': 'center'}),

        html.Div(
            style={"width": "30%", 'display': 'inline-block', },
            children=[
                dcc.Dropdown(
                    id='distribution_column',
                    multi=False,
                    placeholder="Select Distribution column...",
                ),
                dcc.Dropdown(
                    id='categorical_column',
                    multi=False,
                    placeholder="Select Categorical column...",
                ),
                html.Button('Plot Distribution', id='plot_distribution', n_clicks=0, style={
                    'background-color': '#3DAFA8',
                    'color': 'white',
                    'border': 'none',
                    'padding': '10px 20px',
                    'text-align': 'center',
                    'text-decoration': 'none',
                    'display': 'inline-block',
                    'font-size': '16px',
                    'margin': '4px 2px',
                    'cursor': 'pointer',
                    'border-radius': '4px'
                }),
                dcc.Checklist(
                    id='calculate_kde',
                    options=[
                        {'label': 'Calculate KDE', 'value': 'checked'}
                    ],
                    value=[]
                ),
            ]),

        dcc.Graph(id="distributions", figure=go.Figure()),

        html.H3("Correlations", style={'text-align': 'center'}),

        html.Div(
            style={"width": "30%", 'display': 'inline-block', },
            children=[
                dcc.Dropdown(
                    id='correlation_columns',
                    multi=True,
                    placeholder="Select column...",
                ),
                dcc.Dropdown(
                    id='correlation_method',
                    multi=False,
                    options=[
                        {'label': 'Pearson Correlation', 'value': 'pearson'},
                        {'label': 'Spearman Rank Correlation', 'value': 'spearman'},
                        {'label': 'Difference', 'value': 'difference'},
                    ],
                    placeholder="Select method...",
                ),
                html.Button('Calculate Correlation', id='calculate_correlation', n_clicks=0, style={
                    'background-color': '#3DAFA8',
                    'color': 'white',
                    'border': 'none',
                    'padding': '10px 20px',
                    'text-align': 'center',
                    'text-decoration': 'none',
                    'display': 'inline-block',
                    'font-size': '16px',
                    'margin': '4px 2px',
                    'cursor': 'pointer',
                    'border-radius': '4px'
                }),

            ]),

        dcc.Graph(id="correlations", figure=go.Figure()),

        html.H3("Custom Distributions", style={'text-align': 'center'}),

        html.Div(
            style={"width": "30%", 'display': 'inline-block', },
            children=[
                dcc.Dropdown(
                    id='custom_distribution_selection',
                    multi=False,
                    placeholder="Select Custom distribution...",
                ),

                html.Button('Custom Distribution', id='custom_distribution', n_clicks=0, style={
                    'background-color': '#3DAFA8',
                    'color': 'white',
                    'border': 'none',
                    'padding': '10px 20px',
                    'text-align': 'center',
                    'text-decoration': 'none',
                    'display': 'inline-block',
                    'font-size': '16px',
                    'margin': '4px 2px',
                    'cursor': 'pointer',
                    'border-radius': '4px'
                }),

            ]),

        html.Div(id='custom_distribution_output'),

        html.H3("Error Rate", style={'text-align': 'center'}),

        html.Div(
            style={"width": "30%", 'display': 'inline-block', },
            children=[
                dcc.Dropdown(
                    id='error_rate_columns',
                    multi=False,
                    placeholder="Select Error column...",
                ),

                html.Button('Calculate Error Rate', id='calculate_error_rate', n_clicks=0, style={
                    'background-color': '#3DAFA8',
                    'color': 'white',
                    'border': 'none',
                    'padding': '10px 20px',
                    'text-align': 'center',
                    'text-decoration': 'none',
                    'display': 'inline-block',
                    'font-size': '16px',
                    'margin': '4px 2px',
                    'cursor': 'pointer',
                    'border-radius': '4px'
                }),

            ]),
        html.Div(id='error_rate'),
        dcc.Graph(id="test", figure=go.Figure()),


    ])
