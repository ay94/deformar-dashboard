import numpy as np
from layouts import *
import pandas as pd
from dash import Dash, dash_table, dcc, html, callback_context
# from dash.dependencies import Input, Output, State
from layouts import dataset_layout, decision_layout, error_layout, instance_layout
from callbacks import dataset_callback, decision_callback, error_callback, instance_callback


# def create_app():
#     app = start_app()
#
#
#     return app


def start_app():
    app = Dash(__name__)

    app.layout = html.Div([
        dcc.Tabs(id="Tabs", value='dataset', children=[
            dataset_layout.get_layout(),
            decision_layout.get_layout(),
            error_layout.get_layout(),
            instance_layout.get_layout(),
        ]),
    ])
    dataset_obj = dataset_callback.register_dataset_callbacks(app)
    decision_callback.register_decision_callbacks(app, dataset_obj)
    error_callback.register_error_callbacks(app, dataset_obj)
    instance_callback.register_instance_callbacks(app, dataset_obj)
    return app
