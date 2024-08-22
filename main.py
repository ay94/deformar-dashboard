from dash import Dash, dash_table, dcc, html, callback_context
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
from layouts import *
import appSettings
from pathlib import Path

from utils.config_managers import DashboardConfigManager


            
def main():
    CONFIG_PATH = (Path(__file__).parents[1] / "dashboard-config.yaml").resolve()
    
    config_manager = DashboardConfigManager(CONFIG_PATH)
    dev_config = config_manager.development_config    
    app = appSettings.start_app(config_manager)
    app.run_server(debug=dev_config.debug, port=config_manager.development_config.port, dev_tools_hot_reload=True, dev_tools_props_check=True, dev_tools_serve_dev_bundles=True)

if __name__ == '__main__':
    main()