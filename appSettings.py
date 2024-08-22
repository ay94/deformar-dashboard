from dash import Dash, dcc, html, Output, Input
from layouts import load_layout, dataset_layout, decision_layout, performance_layout, instance_layout
from callbacks import load_callback, dataset_callback
from utils.data_managers import DataManager


    
def start_app(config_manager):
    app = Dash(__name__, suppress_callback_exceptions=True)  # Allow dynamic components
    app_config = config_manager.app_config
    server = app.server  # Flask server instance for caching
    variants_data = None
   
    data_manager = DataManager(config_manager, server)
    

    # Initially, only set up the 'load' tab
    # Set up the initial layout with just the tab headers
    tabs = [
        dcc.Tab(label=tab.tab_label, value=tab.tab_value, id=f"tab-{tab.tab_value}", disabled=(tab.tab_value != 'load'))
        for tab in app_config.tabs
    ]

    app.layout = html.Div(
                    children=[
                    dcc.Tabs(id="Tabs", value='load', children=tabs),
                    html.Div(id='tab-content'),  # Placeholder for dynamic content
                    dcc.Interval(id='data-loading-check', interval=1000, n_intervals=0),  # Check every 1 second

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
            dataset_tab = dataset_layout.get_layout(config_manager)
            return dataset_tab
        elif tab == 'decision':
            
            return decision_layout.get_layout()
        elif tab == 'performance':
            
            return performance_layout.get_layout()
        elif tab == 'instance':
            
            return instance_layout.get_layout()
        return html.Div()  # Return an empty div if no tabs match
    
    # Create separate Output for each tab to enable them individually
    @app.callback(
        [Output(f"tab-{tab.tab_value}", "disabled") for tab in app_config.tabs if tab.tab_value != 'load'],
        [Input('data-loading-check', 'n_intervals')]  # Use an interval to periodically check the cache
    )
    def enable_tabs_based_on_cache(n_intervals):
        if data_manager.is_data_loaded():  # Check if all data is loaded
            return [False] * (len(app_config.tabs) - 1)  # Enable all tabs except the load tab
        return [True] * (len(app_config.tabs) - 1)  # Keep other tabs disabled if data is not loaded

    
    variants_data = load_callback.register_callbacks(app, data_manager)
    dataset_callback.register_callbacks(app, variants_data)    
    return app




