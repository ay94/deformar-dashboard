from . import html, dcc

from utils.appUtils import Datasets


def get_layout():

    return dcc.Tab(label='Dataset Characteristics', value="dataset", children=[

        html.Div(
            style={
                "display": "flex",
                "align-items": "center",
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
                    id='exclude_ignored',
                    options=[
                        {'label': 'Exclude IGNORED Tokens', 'value': 'not checked'}
                    ],
                    value=[]
                ),
            ]),

    ])
