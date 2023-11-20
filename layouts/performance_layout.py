from . import html, dcc, go, dash_table


def get_layout():
    return dcc.Tab(label='Performance Analysis', value="performance", children=[
        html.Div([
            html.Div(
                style={
                    "display": "flex",
                    "align-items": "center",
                    'flex-direction': 'column',
                    "justify-content": "center",
                    "height": "30vh",  # Adjust the height as needed
                },
                children=[
                    html.Button('Initialize Performance Tab', id='initialize_error_tab', n_clicks=0, style={
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
                        id='performance_scatter_mode',
                        options=[
                            {'label': 'ScatterMode', 'value': 'group'}
                        ],
                        value=[]
                    ),
                    html.Div(id='initialize_performance_tab')
                ]),
            html.Div(
                style={
                    'display': 'flex',
                    'justify-content': 'center',
                    'align-items': 'center',
                    'height': '50vh'
                },
                children=[
                    html.Button('Compute Metric', id='compute_metric', n_clicks=0, style={
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
                        id='performance_metric',
                        options=[
                            {'label': 'Report', 'value': 'report'},
                            {'label': 'Confusion', 'value': 'confusion'},
                            {'label': 'Errors', 'value': 'errors'}
                        ],
                        placeholder="Select Metric...",
                        style={'width': '150px',
                               'margin': '4px 2px',
                               }
                    ),
                    dcc.Dropdown(
                        id='entity',
                        placeholder="Select Entity...",
                        style={'width': '150px',
                               'margin': '4px 2px',
                               }
                    ),
                ]
            ),

            html.Div(id='performance_container_1',
                     style={
                         'width': '49%',
                         'display': 'inline-block',
                         'margin': '4px 2px',
                     }
                     ),
            html.Div(id='performance_container_2',
                     style={
                         'width': '49%',
                         'display': 'inline-block',
                         'margin': '4px 2px',
                     }
                     ),
            html.Div(
                style={'display': 'flex',
                       'justify-content': 'center',
                       'align-items': 'center',
                       'height': '50vh'},
                children=[

                    html.Button('Filter Error Table', id='filter_error_table', n_clicks=0, style={
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
                        id='filter_column',
                        placeholder="Select Filter column...",
                        style={'width': '250px'}
                    ),

                    dcc.Textarea(
                        id='filter_value',
                        placeholder='Enter Text...',
                        style={'width': 150, 'height': 30,
                               'margin': '4px 2px', 'border': '1px solid #ced4da'}
                    ),

                    html.Button('Reset Error Table', id='reset_error_table', n_clicks=0, style={
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

            dash_table.DataTable(
                id='error_datatable',
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
            html.H3("Performance Scatter Plot", style={'text-align': 'center'}),
            html.Div(
                children=[

                    dcc.Dropdown(
                        id='error_context_color',
                        multi=True,
                        placeholder="Select Color column...",
                        style={'width': '300px'}
                    ),
                    dcc.Dropdown(
                        id='error_x',
                        multi=True,
                        placeholder="Select X column...",
                        style={'width': '300px'}
                    ),
                    dcc.Dropdown(
                        id='error_y',
                        multi=True,
                        placeholder="Select Y column...",
                        style={'width': '300px'}
                    ),

                    html.Button('Generate Errors', id='generate_errors', n_clicks=0, style={
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
                        id='draw_text',
                        options=[
                            {'label': 'Draw Text', 'value': 'draw_text'}
                        ],
                        value=[]
                    ),
                ],
                style={"width": "30%", 'display': 'inline-block'}
            ),
            dcc.Loading(
                id='error_context_loading',
                type='default',
                children=[
                    dcc.Graph(id="error_context_scatter", figure=go.Figure()),
                ]
            ),
            html.Div(id="error_example_ids", style={"display": "none"}),
            html.H3("Error Only Scatter", style={'text-align': 'center'}),
            html.Div(
                children=[

                    dcc.Dropdown(
                        id='error_color',
                        multi=True,
                        placeholder="Select Color column...",
                        style={'width': '300px'}
                    ),

                ],
                style={"width": "30%", 'display': 'inline-block'}
            ),
            dcc.Loading(
                id='error_loading',
                type='default',
                children=[
                    dcc.Graph(id="error_scatter", figure=go.Figure()),
                ]
            ),
            html.H3("Selection Token Ambiguity", style={'text-align': 'center'}),
            html.Div(
                children=[
                    dcc.Dropdown(
                        id='error_tokens',
                        multi=True,
                        placeholder="Select Tokens...",
                        style={'width': '300px'}
                    ),
                    dcc.Loading(
                        id='error_token_ambiguity_loading',
                        type='default',
                        children=[
                            dcc.Graph(id="error_token_ambiguity", figure=go.Figure().update_layout(
                                title="Selected Token Ambiguity"
                            )),
                        ]
                    ),
                ],
                style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}
            ),
            html.H3("Selection Token Similarity Matrix", style={'text-align': 'center'}),
            html.Div(
                children=[
                    dcc.Dropdown(
                        id='error_similarity_tokens',
                        multi=True,
                        placeholder="Select Tokens...",
                        style={'width': '300px'}
                    ),
                    html.Button('Compute Error Similarity Matrix', id='compute_error_similarity_matrix', n_clicks=0, style={
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
                    dcc.Loading(
                        id='error_similarity_matrix_loading',
                        type='default',
                        children=[
                            dcc.Graph(id="error_similarity_matrix", figure=go.Figure().update_layout(
                                title="Selected Tokens Similarity Matrix"
                            )),
                        ]
                    ),
                    html.Div(id='error_similarity_status')
                ],
                style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}
            ),
        ])
    ])



