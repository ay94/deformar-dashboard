import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html

from callbacks import cross_component_callback, decision_callback, load_callback, instance_callback
from dataManager import DataManager
from layouts.tabs import (
                    cross_component_layout,
                    decision_layout,
                    instance_layout,
                    load_layout
                    )


def start_app(config_manager):
    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
    )  # Allow dynamic components

    app_config = config_manager.app_config
    server = app.server  # Flask server instance for caching
    variants_data = None

    data_manager = DataManager(config_manager, server)

    # Initially, only set up the 'load' tab
    # Set up the initial layout with just the tab headers
    tabs = [
        dcc.Tab(
            label=tab.tab_label,
            value=tab.tab_value,
            id=f"tab-{tab.tab_value}",
            disabled=(tab.tab_value != "load"),
        )
        for tab in app_config.tabs
    ]

    app.layout = html.Div(
        children=[
            dcc.Tabs(id="Tabs", value="load", children=tabs),
            html.Div(id="tab-content"),  # Placeholder for dynamic content
            dcc.Store(
                id="tab-store", storage_type="local"
            ),  # Initialize with default tab
            dcc.Interval(
                id="data-loading-check", interval=1000, n_intervals=0
            ),  # Check every 1 second
            dcc.Store(id="decision_store"),
            dcc.Store(id="measure_store"),
        ]
    )
    tab_layouts = {
        "load": load_layout.get_layout(config_manager),
        "quantitative": cross_component_layout.get_layout(config_manager),
        "qualitative": decision_layout.get_layout(config_manager),
        "instance": instance_layout.get_layout(config_manager),
    }

    @app.callback(Output("tab-content", "children"), Input("Tabs", "value"))
    def render_tab_content(selected_tab):

        tab_content = tab_layouts.get(selected_tab, lambda: html.Div("Tab not found"))
        return tab_content

    @app.callback(
        Output("tab-store", "data"),
        Input("Tabs", "value"),
        prevent_initial_call=True,  # Prevent running on app initialization
    )
    def update_current_tab(selected_tab):
        return {"current_tab": selected_tab}

    @app.callback(
        Output("Tabs", "value"),
        Input("current-tab-store", "data"),
        prevent_initial_call=True,
    )
    def set_initial_tab(stored_data):
        if stored_data:
            return stored_data.get("current_tab", "load")
        return "load"

    @app.callback(
        [
            Output(f"tab-{tab.tab_value}", "disabled")
            for tab in app_config.tabs
            if tab.tab_value != "load"
        ],
        Input("data-loading-check", "n_intervals"),
    )
    def enable_tabs_based_on_cache(n_intervals):
        if data_manager.is_any_variant_loaded():
            return [False] * (
                len(app_config.tabs) - 1
            )  # Enable all tabs except the load tab
        return [True] * (
            len(app_config.tabs) - 1
        )  # Keep other tabs disabled if data is not loaded

    variants_data = load_callback.register_callbacks(app, data_manager)
    cross_component_callback.register_callbacks(app, variants_data, config_manager)
    decision_callback.register_callbacks(app, variants_data)
    instance_callback.register_callbacks(app, variants_data)
    return app
