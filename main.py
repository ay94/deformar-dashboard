from dash import Dash, dash_table, dcc, html, callback_context
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
from layouts import *
import appConfig


def main():
    app = appConfig.start_app()
    app.run_server(debug=True)

if __name__ == '__main__':
    main()