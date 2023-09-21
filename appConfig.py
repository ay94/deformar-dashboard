import numpy as np
from layouts import *
import pandas as pd
from dash import Dash, dash_table, dcc, html, callback_context
# from dash.dependencies import Input, Output, State
from layouts import load_layout, dataset_layout, decision_layout, performance_layout, instance_layout
from callbacks import load_callback, dataset_callback, decision_callback, performance_callback, instance_callback


# def create_app():
#     app = start_app()
#
#
#     return app


def start_app():
    app = Dash(__name__)

    app.layout = html.Div([
        dcc.Tabs(id="Tabs", value='load', children=[
            load_layout.get_layout(),
            dataset_layout.get_layout(),
            decision_layout.get_layout(),
            performance_layout.get_layout(),
            instance_layout.get_layout(),
        ]),
    ])
    dataset_obj = load_callback.register_load_callbacks(app)
    dataset_callback.register_dataset_callbacks(app, dataset_obj)
    decision_callback.register_decision_callbacks(app, dataset_obj)
    performance_callback.register_error_callbacks(app, dataset_obj)
    instance_callback.register_instance_callbacks(app, dataset_obj)
    return app
