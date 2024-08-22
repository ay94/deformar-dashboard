import numpy as np
from layouts import *
import pandas as pd
from dash import Dash, dash_table, dcc, html, callback_context, Output, Input, State
# from dash.dependencies import Input, Output, State
from layouts import load_layout, dataset_layout, decision_layout, performance_layout, instance_layout
from callbacks import load_callback, dataset_callback, decision_callback, performance_callback, instance_callback

from utils.data_managers import DataManager

    
def start_app(config_manager):
    app = Dash(__name__, suppress_callback_exceptions=True)  # Allow dynamic components
    app_config = config_manager.app_config
    server = app.server  # Flask server instance for caching
    variants_data = None
   
    data_manager = DataManager(config_manager, server)
    

    # Initially, only set up the 'load' tab
    # Set up the initial layout with just the tab headers
    tabs = [dcc.Tab(label=tab.tab_label, value=tab.tab_value) for tab in app_config.tabs]

    app.layout = html.Div(
                    children=[
                    dcc.Tabs(id="Tabs", value='load', children=tabs),
                    html.Div(id='tab-content'),  # Placeholder for dynamic content
                    
                    ]
                )
    
    @app.callback(
        Output('tab-content', 'children'),
        [
            Input('Tabs', 'value'),
        ],
    )
    def render_tab_content(tab, ):
        # Decide what content to load based on the selected tab
        if tab == 'load':
            load_tab = load_layout.get_layout(config_manager)
            return load_tab
        elif tab == 'dataset':            
            return dataset_layout.get_layout(config_manager)
        elif tab == 'decision':
            # decision_callback.register_decision_callbacks(dataset_obj, app)
            return decision_layout.get_layout()
        elif tab == 'performance':
            # performance_callback.register_error_callbacks(dataset_obj, app)
            return performance_layout.get_layout()
        elif tab == 'instance':
            # instance_callback.register_instance_callbacks(dataset_obj, app)
            return instance_layout.get_layout()
        return html.Div()  # Return an empty div if no tabs match
    variants_data = load_callback.register_callbacks(app, data_manager)
    dataset_callback.register_callbacks(app, variants_data)    
    return app




