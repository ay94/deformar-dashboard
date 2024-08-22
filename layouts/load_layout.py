from . import html, dcc

from utils.appUtils import Datasets

from plotly import express as px

from dash import dcc, html
from utils.load_layout_managers import LoadTabLayout

def get_layout(config_manager):
    tab_layout = LoadTabLayout(config_manager)   
    return tab_layout.render()
    # variants = config_manager.app_config.variants  # Dropdown options for variants.
    # layout = html.Div([
    #             html.Div(
    #                 style={
    #                     'display': 'flex',
    #                     'flex-direction': 'column',  # Align children in a column
    #                     'align-items': 'center',     # Center-align items horizontally
    #                     'justify-content': 'center', # Center-align items vertically
    #                     'height': '100vh',           # Optionally set the height to full viewport height
    #                     'padding': '20px'            # Add padding around the entire container
    #                 },
    #                 children=[
    #                     html.Div(
    #                         style={
    #                             'display': 'flex',
    #                             'justify-content': 'center',
    #                             'width': '100%'  # Ensure this div takes the full width
    #                         },
    #                         children=[
    #                             dcc.Dropdown(
    #                                 options=[{'label': variant, 'value': variant} for variant in variants],
    #                                 id='variant_name',
    #                                 placeholder="Choose Experiment Variant...",
    #                                 style={'width': '300px', 'margin-right': '10px'}
    #                             ),
    #                             html.Button('Load Variant', id='load_variant', n_clicks=0, style={
    #                                 'background-color': '#3DAFA8', 'color': 'white', 'margin-right': '5px',
    #                                 'cursor': 'pointer', 'border-radius': '4px', 'border-color': 'white'
    #                             }),
    #                             html.Button('Load Data', id='load_data', n_clicks=0, style={
    #                                 'background-color': '#3DAFA8', 'color': 'white',
    #                                 'cursor': 'pointer', 'border-radius': '4px', 'border-color': 'white'
    #                             }),
    #                             html.Button('Clear Cache', id='clear_cache', n_clicks=0, style={
    #                                 'background-color': '#3DAFA8', 'color': 'white',
    #                                 'cursor': 'pointer', 'border-radius': '4px', 'border-color': 'white'
    #                             })
    #                         ]
    #                     ),
    #                     dcc.Loading(
    #                         id="data_loading_container",
    #                         type="default",
    #                         children=[
    #                             html.Div(id='load_message', style={'margin-top': '20px', 'color': '#28a745'}),  # Style for the message
                                
    #                             html.Div(id='cache_status', style={'margin-top': '20px', 'color': '#28a745'}),  # Style for the message
    #                             html.Div(id='data_status_table', style={'padding': '20px'})  # Style for the display data section
    #                         ],
    #                         fullscreen=False,  # Set fullscreen to False
    #                     ),
    #                 ]
    #             ),
                
    #         ])
    # return layout

#     data_dir = config_manager.data_dir  # Used for logging or displaying the data directory.
#     layout = html.Div([
#         html.Div(
#             style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'height': '10vh'},
#             children=[
#                 dcc.Dropdown(
#                     options=[{'label': variant, 'value': variant} for variant in variants],
#                     id='variant_name',
#                     placeholder="Choose Experiment Variant...",
#                     style={'width': '300px', 'margin': '4px 2px'}
#                 ),
#                 html.Button('Load variant', id='load_variant', n_clicks=0, style={
#                     'background-color': '#3DAFA8', 'color': 'white', 'border': 'none',
#                     'padding': '10px 20px', 'text-align': 'center', 'text-decoration': 'none',
#                     'display': 'inline-block', 'font-size': '16px', 'margin': '4px 2px',
#                     'cursor': 'pointer', 'border-radius': '4px'
#                 }),
#                 html.Button('Load Data', id='load_data', n_clicks=0, style={
#                     'background-color': '#3DAFA8', 'color': 'white', 'border': 'none',
#                     'padding': '10px 20px', 'text-align': 'center', 'text-decoration': 'none',
#                     'display': 'inline-block', 'font-size': '16px', 'margin': '4px 2px',
#                     'cursor': 'pointer', 'border-radius': '4px'
#                 }),
#                 html.Div(id='load_message')
#             ]
            
#         ),
#         html.Div(id='display_data'),
        
#     ])
#     return layout

# # def get_layout(config_manager):
# #     variants = config_manager.app_config.variants  # Assuming these are your dropdown options.
# #     data_dir = config_manager.data_dir  # This might be used to display or log the data directory.

# #     return html.Div([
# #         html.Div(
# #             style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'height': '10vh'},
# #             children=[
# #                 dcc.Dropdown(
# #                     options=[{'label': variant, 'value': variant} for variant in variants],
# #                     id='variant_name',
# #                     placeholder="Choose Experiment Variant...",
# #                     style={'width': '300px', 'margin': '4px 2px'}
# #                 ),
# #                 html.Button('Load Dataset', id='load_dataset', n_clicks=0, style={
# #                     'background-color': '#3DAFA8', 'color': 'white', 'border': 'none',
# #                     'padding': '10px 20px', 'text-align': 'center', 'text-decoration': 'none',
# #                     'display': 'inline-block', 'font-size': '16px', 'margin': '4px 2px',
# #                     'cursor': 'pointer', 'border-radius': '4px'
# #                 }),
# #             ]
# #         ),
# #         html.Div(id='dataset_loaded_message')
# #     ])

# # def get_layout():
# #     dataset_names = list(Datasets.keys())
# #     return dcc.Tab(label='Load Dataset', value="load", children=[
# #         html.Div(
# #             style={'display': 'flex',
# #                    'justify-content': 'center',
# #                    'align-items': 'center',
# #                    'height': '10vh'
# #                    },
# #             children=[

# #                 html.Button('Load Dataset', id='load_dataset', n_clicks=0, style={
# #                     'background-color': '#3DAFA8',
# #                     'color': 'white',
# #                     'border': 'none',
# #                     'padding': '10px 20px',
# #                     'text-align': 'center',
# #                     'text-decoration': 'none',
# #                     'display': 'inline-block',
# #                     'font-size': '16px',
# #                     'margin': '4px 2px',
# #                     'cursor': 'pointer',
# #                     'border-radius': '4px'
# #                 }),
# #                 dcc.Dropdown(
# #                     dataset_names,
# #                     id='dataset_name',
# #                     placeholder="Choose Dataset...",
# #                     style={'width': '250px', 'margin': '4px 2px', }
# #                 ),

# #                 dcc.Dropdown(
# #                     id='model_name',
# #                     placeholder="Choose Model...",
# #                     style={'width': '200px', 'margin': '4px 2px', }
# #                 ),

# #                 dcc.Dropdown(
# #                     id='split',
# #                     placeholder="Choose Split...",
# #                     style={'width': '150px', 'margin': '4px 2px', }
# #                 ),

# #                 html.Button('Initialize Model', id='initialize_model', n_clicks=0, style={
# #                     'background-color': '#3DAFA8',
# #                     'color': 'white',
# #                     'border': 'none',
# #                     'padding': '10px 20px',
# #                     'text-align': 'center',
# #                     'text-decoration': 'none',
# #                     'display': 'inline-block',
# #                     'font-size': '16px',
# #                     'margin': '4px 2px',
# #                     'cursor': 'pointer',
# #                     'border-radius': '4px'
# #                 }),

# #             ]),
# #         dcc.Loading(
# #             id="data_loading_container",
# #             type="default",
# #             children=[
# #                 html.Div(
# #                     style={
# #                         'display': 'flex',
# #                         'flex-direction': 'column',  # Stack elements below each other
# #                         'justify-content': 'flex-start',  # Align items to the top of the container
# #                         'align-items': 'flex-start',  # Align items to the start of the container horizontally
# #                         'height': '10vh'
# #                     },
# #                     children=[
# #                         html.Div(id='dataset_prompt'),
# #                         html.Div(id='load_dataset_prompt'),
# #                         html.Div(id='initialized_dataset_prompt')
# #                     ]),
# #             ],
# #             fullscreen=False,  # Set fullscreen to False
# #         ),

# #         html.Div(
# #             style={
# #                 "display": "flex",
# #                 "align-items": "center",
# #                 "justify-content": "center",
# #                 "height": "30vh",  # Adjust the height as needed
# #             },
# #             children=[
# #                 html.Button('View Dataset Table', id='view_dataset', n_clicks=0, style={
# #                     'background-color': '#3DAFA8',
# #                     'color': 'white',
# #                     'border': 'none',
# #                     'padding': '10px 20px',
# #                     'text-align': 'center',
# #                     'text-decoration': 'none',
# #                     'display': 'inline-block',
# #                     'font-size': '16px',
# #                     'margin': '4px 2px',
# #                     'cursor': 'pointer',
# #                     'border-radius': '4px'
# #                 }),

# #                 html.Button('Hide Dataset Table', id='hide_dataset', n_clicks=0, style={
# #                     'background-color': '#3DAFA8',
# #                     'color': 'white',
# #                     'border': 'none',
# #                     'padding': '10px 20px',
# #                     'text-align': 'center',
# #                     'text-decoration': 'none',
# #                     'display': 'inline-block',
# #                     'font-size': '16px',
# #                     'margin': '4px 2px',
# #                     'cursor': 'pointer',
# #                     'border-radius': '4px'
# #                 }),

# #             ]),
# #         html.Div(id='dataset_table_container'),
# #         # html.Div([
# #         #     dcc.Graph(id="test", figure=px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])),
# #         #     html.Button('Download PDF', id='download', n_clicks=0, style={
# #         #         'background-color': '#3DAFA8',
# #         #         'color': 'white',
# #         #         'border': 'none',
# #         #         'padding': '10px 20px',
# #         #         'text-align': 'center',
# #         #         'text-decoration': 'none',
# #         #         'display': 'inline-block',
# #         #         'font-size': '16px',
# #         #         'margin': '4px 2px',
# #         #         'cursor': 'pointer',
# #         #         'border-radius': '4px'
# #         #     }),
# #         #     html.Br(),
# #         #     html.A("Download PDF", id='download-link')
# #         # ])

# #     #     Pattern - Matching Callbacks(Dash 0.48.0 and later):
# #     # Dash
# #     # introduced
# #     # pattern - matching
# #     # callbacks
# #     # that
# #     # allow
# #     # callbacks
# #     # to
# #     # match
# #     # components
# #     # based
# #     # on
# #     # the
# #     # properties
# #     # of
# #     # the
# #     # component’s
# #     # ID.This
# #     # could
# #     # be
# #     # a
# #     # powerful
# #     # feature
# #     # to
# #     # handle
# #     # multiple
# #     # instances
# #     # of
# #     # similar
# #     # components.
# #     #     app.layout = html.Div([
# #     #     ...
# #     #     html.Button("Download PDF", id={'type': 'download-button', 'index': 1}),
# #     #     html.A("Download PDF", id={'type': 'download-link', 'index': 1}),
# #     #     ...
# #     #     html.Button("Download PDF", id={'type': 'download-button', 'index': 2}),
# #     #     html.A("Download PDF", id={'type': 'download-link', 'index': 2}),
# #     #     ...
# #     # ])
# #     #
# #     #                  @ app.callback(
# #     #     Output({'type': 'download-link', 'index': MATCH}, 'href'),
# #     #     Input({'type': 'download-button', 'index': MATCH}, 'n_clicks'),
# #     #     State({'type': 'graph', 'index': MATCH}, 'figure'),
# #     #     prevent_initial_call=True
# #     # )
# #     #
# #     # def update_download_link(n_clicks, current_figure):
# #     #     ...

# #     ])
