from dash import html, dcc
from dash import html, dash_table
import plotly.graph_objs as go


from utils.layout_managers import generate_dropdown_options, CustomButton


class LoadTabLayout:
    def __init__(self, config_manager):
        self.variants = config_manager.app_config.variants

    def render(self):
        dropdown = dcc.Dropdown(
            options=generate_dropdown_options(self.variants),
            id='variant_name',
            placeholder="Choose Experiment Variant...",
            style={'width': '300px', 'margin-right': '10px'}
        )
        load_button = CustomButton('Load Variant', 'load_variant').render()
        load_data_button = CustomButton('Load Data', 'load_data').render()
        clear_cache_button = CustomButton('Clear Cache', 'clear_cache').render()

        layout = html.Div([
            html.Div(style={
                'display': 'flex',
                'flex-direction': 'column',
                'align-items': 'center',
                'justify-content': 'center',
                'height': '100vh',
                'padding': '20px'
            }, children=[
                html.Div(style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'width': '100%'}, children=[
                    dropdown, load_button, load_data_button, clear_cache_button
                ]),
                dcc.Loading(
                    id="data_loading_container",
                    type="default",
                    children=[
                        html.Div(id='load_message', style={'margin-top': '20px', 'color': '#28a745', 'textAlign': 'center'}),
                        html.Div(id='cache_status', style={'margin-top': '20px', 'color': '#28a745'}),
                        html.Div(id='data_status_table', style={'padding': '20px'})
                    ],
                    fullscreen=False,
                ),
            ])
        ])
        return layout



