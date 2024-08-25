from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from dash import  dcc, html, no_update

import plotly.graph_objs as go

from utils.decision_tab_mangers import DecisionTabManager



def register_callbacks(app, variants_data):
    tab_manager = DecisionTabManager(variants_data)