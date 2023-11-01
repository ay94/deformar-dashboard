from . import html, dcc, go, dash_table


def get_layout():
    return dcc.Tab(label='Decision Boundary', value="decision", children=[
        html.Div(

            children=[

                html.Div(
                    style={
                        "display": "flex",
                        "align-items": "center",
                        "justify-content": "center",
                        "height": "30vh",  # Adjust the height as needed
                    },
                    children=[
                        html.Button('Initialize Decision Tab', id='initialize_decision_tab', n_clicks=0, style={
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
                            id='scatter_mode',
                            options=[
                                {'label': 'ScatterMode', 'value': 'group'}
                            ],
                            value=[]
                        ),
                ]),
                dash_table.DataTable(
                    id='selector_datatable',
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

                html.Div(id="save_decision_points", style={"display": "none"}),
                html.Div(id="save_measure_points", style={"display": "none"}),
                html.H3("Selector Scatter Plot", style={'text-align': 'center'}),
                dcc.Loading(
                    id='selector_loading',
                    type='default',
                    children=[
                        dcc.Graph(id="selector_scatter", figure=go.Figure()),
                    ]
                )

            ]),
        html.H3("Decision Boundary Scatter Plot", style={'text-align': 'center'}),
        html.Div(
            style={"width": "30%", 'display': 'inline-block', },
            children=[
                dcc.Dropdown(
                    id='decision_columns',
                    multi=True,
                    placeholder="Select Color column...",
                ),
                html.Button('Generate Decision Plot', id='generate_decision', n_clicks=0, style={
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
                html.Button('Reset Selection', id='reset_decision', n_clicks=0, style={
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

        dcc.Loading(
            id='decision_loading',
            type='default',
            children=[
                dcc.Graph(id="decision_scatter", figure=go.Figure()),
            ]
        ),
        html.H3("Decision Boundary Measures Scatter Plot", style={'text-align': 'center'}),
        html.Div(
            [
                dcc.Dropdown(
                    id='measure_columns',
                    multi=True,
                    placeholder="Select Color column...",
                ),
                dcc.Dropdown(
                    id='measure_x',
                    multi=True,
                    placeholder="Select X column...",
                ),
                dcc.Dropdown(
                    id='measure_y',
                    multi=True,
                    placeholder="Select Y column...",
                ),
                html.Button('Generate Measure Plot', id='generate_measure', n_clicks=0, style={
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
                html.Button('Reset Selection', id='reset_measure', n_clicks=0, style={
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
                    id='include_ignored',
                    options=[
                        {'label': 'Include IGNORED Tokens', 'value': 'checked'}
                    ],
                    value=[]
                ),
            ],
            style={"width": "30%", 'display': 'inline-block', }
        ),
        dcc.Loading(
            id='measure_loading',
            type='default',
            children=[
                dcc.Graph(id="measure_scatter", figure=go.Figure()),
            ]
        ),
        html.H3("Centroid Scatter Plot", style={'text-align': 'center'}),
        html.Div(
            [
                dcc.Dropdown(
                    id='centroid_columns',
                    multi=True,
                    placeholder="Select Color column...",
                ),

                dcc.Dropdown(
                    id='centroid_cluster',
                    placeholder="Select Centroid",
                ),

                html.Button('Generate Centroid Plot', id='generate_centroid', n_clicks=0, style={
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

            ],
            style={"width": "30%", 'display': 'inline-block', }
        ),
        dcc.Loading(
            id='centroid_loading',
            type='default',
            children=[
                dcc.Graph(id="centroid_scatter", figure=go.Figure()),
            ]
        ),
        html.H3("Selection Tag Proportion", style={'text-align': 'center'}),
        dcc.Loading(
            id='tag_proportion_loading',
            type='default',
            children=[
                dcc.Graph(id="selection_tag_proportion", figure=go.Figure()),
            ]
        ),
        html.H3("Selection Datatable", style={'text-align': 'center'}),
        html.Div(id='selection_datatable',
                 style={'text-align': 'center'},
                 children=[
                     # Default text when data is not available
                     "No Data Selected.",
        ]),
        html.H3("Selection Token Ambiguity", style={'text-align': 'center'}),
        dcc.Loading(
            id='selection_token_ambiguity_loading',
            type='default',
            children=[
                dcc.Graph(id="selection_token_ambiguity", figure=go.Figure()),
            ]
        ),
        html.H3("Training Impact", style={'text-align': 'center'}),
        html.Div(
            style={
                'display': 'flex',
                'justify-content': 'center',
                'align-items': 'center',
                'height': '100vh'
            },
            children=[
                html.Button('Show Impact', id='show_impact', n_clicks=0, style={
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
                dcc.Dropdown(
                    id='impact_view',
                    options=[
                        {'label': 'Attention Scores', 'value': 'attention_score'},
                        {'label': 'Weights', 'value': 'weights'},
                    ],
                    placeholder="Select View...",
                    style={'width': '150px',
                           'margin': '4px 2px',
                           }
                ),
                dcc.Loading(
                    id='impact_loading',
                    type='default',
                    children=[
                        dcc.Graph(id="impact_plot", figure=go.Figure()),
                    ]
                ),
            ]
        ),

    ])
