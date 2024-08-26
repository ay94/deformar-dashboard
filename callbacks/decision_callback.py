from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from dash import  dcc, html, no_update

import plotly.graph_objs as go

from utils.decision_tab_mangers import DecisionTabManager



def register_callbacks(app, variants_data):
    tab_manager = DecisionTabManager(variants_data)
    
    @app.callback(
        Output("training_graph", "children"),    
        [
            Input('variant_selector', 'value'),
        ]
    )
    def training_decision_boundary(variant):
        data = tab_manager.get_training_data(variant)
        if data is None:
            return html.Div(
                "Please extract training data to be able to view it.",
                className='prompt-message'
            )
        return html.Div(
                "Please extract training data to be able to view it.",
                className='prompt-message'
            )
    @app.callback(
        [
            Output("correlation_heatmap", "figure"),
            Output("correlation_heatmap", "style")
        ],
        [
            Input('view_decision_boundary', 'n_clicks'),
        ],
        [
            State('decision_correlation_coefficient', 'value'),
            State('variant_selector', 'value')
        ]
    )
    def generate_matrix(n_clicks, correlation_method, variant):
        # Default to no update
        matrix_fig = no_update
        matrix_style = {'display': 'none'}

        # Check if button click is valid
        if n_clicks is not None and n_clicks > 0:
            # Calculate the initial correlation and scatter plot figures
            correlation_method = correlation_method if correlation_method else 'Pearson'
            matrix_fig = tab_manager.generate_matrix(
                variant=variant,
                correlation_method=correlation_method,
            )
            matrix_style = {'display': 'inline-block'}
        if matrix_fig is None:
            matrix_fig = {}  # Ensure a valid empty dictionary if None is returned
        # Return figures and styles
        return matrix_fig, matrix_style
    
    @app.callback(
        [
            Output("decision_scatter", "figure"),
            Output("decision_scatter", "style")
        ],
        [
            Input('view_decision_boundary', 'n_clicks'),
        ],
        [
            State('variant_selector', 'value'),
            State('decision_type', 'value'),
            State('decision_columns', 'value'),
        ]
    )
    def generate_decision_plot(n_clicks, variant, decision_type, color_column):
        # Default to no update
        decision_fig = no_update
        decision_style = {'display': 'none'}

        # Check if button click is valid
        if n_clicks is not None and n_clicks > 0:
            # Calculate the initial correlation and scatter plot figures
            print("I am decsion type drop", decision_type)
            decision_type = decision_type if decision_type else 'Fine Tuned Model'
            color_column = color_column if color_column else 'True Labels'
            print("I am decsion type drop", decision_type)
            decision_fig = tab_manager.generate_scatter_plot(
                variant=variant,
                decision_type=decision_type,
                color_column=color_column
            )
            decision_style = {'display': 'inline-block'}
        if decision_fig is None:
            decision_fig = {}  # Ensure a valid empty dictionary if None is returned
        # Return figures and styles
        return decision_fig, decision_style
    
    @app.callback(
        [
            Output("decision_scatter", "figure"),
            Output("decision_scatter", "style")
        ],
        [
            Input('view_decision_boundary', 'n_clicks'),
        ],
        [
            State('variant_selector', 'value'),
            State('decision_type', 'value'),
            State('decision_columns', 'value'),
        ]
    )
    def generate_decision_plot(n_clicks, variant, decision_type, color_column):
        # Default to no update
        decision_fig = no_update
        decision_style = {'display': 'none'}

        # Check if button click is valid
        if n_clicks is not None and n_clicks > 0:
            # Calculate the initial correlation and scatter plot figures
            print("I am decsion type drop", decision_type)
            decision_type = decision_type if decision_type else 'Fine Tuned Model'
            color_column = color_column if color_column else 'True Labels'
            print("I am decsion type drop", decision_type)
            decision_fig = tab_manager.generate_scatter_plot(
                variant=variant,
                decision_type=decision_type,
                color_column=color_column
            )
            decision_style = {'display': 'inline-block'}
        if decision_fig is None:
            decision_fig = {}  # Ensure a valid empty dictionary if None is returned
        # Return figures and styles
        return decision_fig, decision_style
    
    