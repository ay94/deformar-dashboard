import json

import dash
import plotly.graph_objs as go
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from layouts.managers.layout_managers import (get_input_trigger,
                                              process_selection)
from managers.tabs.decision_tab_mangers import DecisionTabManager


def register_callbacks(app, variants_data):
    tab_manager = DecisionTabManager(variants_data)

    @app.callback(
        Output("training_graph", "children"),
        [
            Input("variant_selector", "value"),
            Input("view_training_decision_boundary", "n_clicks"),
        ],
    )
    def training_decision_boundary(variant, n_clicks):
        if n_clicks is None or n_clicks == 0:
            return html.Div(
                "Please extract training data or click the button.",
                className="prompt-message",
            )
        fig = tab_manager.get_training_data(variant)
        if fig is None:
            return html.Div(
                "Please extract training data or click the button.",
                className="prompt-message",
            )
        return dcc.Graph(figure=fig)

    @app.callback(
        [
            Output("correlation_heatmap", "figure"),
            Output("correlation_heatmap", "style"),
        ],
        [
            Input("view_decision_boundary", "n_clicks"),
        ],
        [
            State("decision_correlation_coefficient", "value"),
            State("variant_selector", "value"),
        ],
    )
    def generate_matrix(n_clicks, correlation_method, variant):
        # Default to no update
        matrix_fig = no_update
        matrix_style = {"width": "100%", "height": "500px", "display": "none"}

        # Check if button click is valid
        if n_clicks is not None and n_clicks > 0:
            # Calculate the initial correlation and scatter plot figures
            correlation_method = correlation_method if correlation_method else "Pearson"
            matrix_fig = tab_manager.generate_matrix(
                variant=variant,
                correlation_method=correlation_method,
            )
            matrix_style = {
                "width": "100%",
                "height": "500px",
                "display": "inline-block",
            }
        if matrix_fig is None:
            matrix_fig = {}  # Ensure a valid empty dictionary if None is returned
        # Return figures and styles
        return matrix_fig, matrix_style

    @app.callback(
        [Output("decision_scatter", "figure"), Output("decision_scatter", "style")],
        [
            Input("view_decision_boundary", "n_clicks"),
            Input("measure_store", "data"),
        ],
        [
            State("variant_selector", "value"),
            State("decision_type", "value"),
            State("decision_columns", "value"),
        ],
    )
    def generate_decision_plot(
        n_clicks, measureSelection, variant, decision_type, color_columns
    ):

        fig = no_update
        style = {"width": "100%", "height": "500px", "display": "none"}

        if n_clicks is not None and n_clicks > 0:

            decision_type = decision_type if decision_type else "Fine Tuned Model"
            color_column = "True Labels"
            symbol_column = None
            selection_ids = None
            if color_columns:
                color_column = color_columns[0]
                if len(color_columns) > 1:
                    symbol_column = color_columns[1]
            selection_ids = process_selection(measureSelection)
            fig = tab_manager.generate_decision_plot(
                variant=variant,
                decision_type=decision_type,
                color_column=color_column,
                symbol_column=symbol_column,
                selection_ids=selection_ids,
            )
            style = {"width": "100%", "height": "600px", "display": "inline-block"}
        if fig is None:
            fig = {}

        return fig, style

    @app.callback(
        [Output("measure_scatter", "figure"), Output("measure_scatter", "style")],
        [
            Input("view_decision_boundary", "n_clicks"),
            Input("correlation_heatmap", "clickData"),
            Input("decision_store", "data"),
        ],
        [
            State("variant_selector", "value"),
            State("decision_columns", "value"),
        ],
    )
    def generate_measure_plot(
        n_clicks, clickData, decisionSelection, variant, color_columns
    ):
        if n_clicks is None or n_clicks == 0:
            return no_update, {"width": "100%", "height": "500px", "display": "none"}

        style = {"width": "100%", "height": "500px", "display": "inline-block"}
        color_column = color_columns[0] if color_columns else "True Labels"
        symbol_column = (
            color_columns[1] if color_columns and len(color_columns) > 1 else None
        )

        x_column = "Pre X"
        y_column = "Pre Y"
        selection_ids = None
        if clickData and "points" in clickData and len(clickData["points"]) > 0:
            x_column = clickData["points"][0].get("x", "X")
            y_column = clickData["points"][0].get("y", "Y")

        selection_ids = process_selection(decisionSelection)

        fig = tab_manager.generate_measure_plot(
            variant=variant,
            x_column=x_column,
            y_column=y_column,
            color_column=color_column,
            symbol_column=symbol_column,
            selection_ids=selection_ids,
        )

        return fig, style

    @app.callback(
        Output("decision_store", "data"),
        [
            Input("decision_scatter", "selectedData"),
            Input("clear_decision", "n_clicks"),
        ],
        prevent_initial_call=True,
    )
    def store_decision_selection(selectedData, _):
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update

        trigger_id = get_input_trigger(ctx)
        if trigger_id == "clear_decision":
            return None  # Clear the store
        elif trigger_id == "decision_scatter":
            if selectedData and "points" in selectedData:
                extracted_data = [
                    {"Global Id": point["customdata"][0]}
                    for point in selectedData["points"]
                    if "customdata" in point and len(point["customdata"]) > 0
                ]
                return json.dumps(extracted_data, indent=2)
            return no_update
        return no_update

    @app.callback(
        Output("measure_store", "data"),
        [
            Input("measure_scatter", "selectedData"), 
            Input("clear_measure", "n_clicks")
        ],
        prevent_initial_call=True,
    )
    def store_measure_selection(selectedData, _):
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update

        trigger_id = get_input_trigger(ctx)
        if trigger_id == "clear_measure":
            return None  # Clear the store
        elif trigger_id == "measure_scatter":
            if selectedData and "points" in selectedData:
                extracted_data = [
                    {"Global Id": point["customdata"][0]}
                    for point in selectedData["points"]
                    if "customdata" in point and len(point["customdata"]) > 0
                ]
                return json.dumps(extracted_data, indent=2)
            return no_update
        return no_update

    @app.callback(
        Output("selection_tag_container", "children"),
        [
            Input("measure_store", "data"),  # Correctly using 'data' property
            Input("decision_store", "data"),
        ],
        [
            State("variant_selector", "value"),
        ],
    )
    def generate_selection_output(measureSelection, decisionSelection, variant):
        # Convert selections to lists of IDs
        measure_selection_ids = process_selection(measureSelection)
        decision_selection_ids = process_selection(decisionSelection)

        # Determine which selection IDs to use
        if decision_selection_ids:  # Use decision selection if available
            selection_ids = decision_selection_ids
        elif measure_selection_ids:  # Fallback to measure selection
            selection_ids = measure_selection_ids
        else:
            # No data selected for either case
            return html.Div(
                "No data selected from measure or decision plots.",
                className="prompt-message",
            )

        # Generate the plot or figure
        fig = tab_manager.generate_tag_proportion(
            variant=variant, selection_ids=selection_ids
        )
        if isinstance(fig, go.Figure):
            return dcc.Graph(figure=fig, style={"width": "100%", "height": "500px"})

    @app.callback(
        Output("centroid_matrix_container", "children"),
        [
            Input("variant_selector", "value"),
        ],
    )
    def generate_centroid_matrix(variant):
        centroid_matrix = tab_manager.generate_centroid_matrix(variant)

        if centroid_matrix is None:
            return html.Div(
                "Please select a type of results and click 'View Results' to view data.",
                className="prompt-message",
            )
        return dcc.Graph(
            figure=centroid_matrix, style={"width": "100%", "height": "100%"}
        )

    @app.callback(
        [
            Output("attention_similarity_matrix_container", "children"),
            Output("attention_weight_similarity_container", "children"),
        ],
        Input("variant_selector", "value"),
    )
    def view_training_impact(variant):
        attention_weights, attention_matrices = tab_manager.generate_training_impact(
            variant
        )

        if attention_weights is None and attention_matrices is None:
            return html.Div(
                "No Attention Weights Available", className="prompt-message"
            ), html.Div("No Attention Matrices Available", className="prompt-message")

        attention_weights_graph = dcc.Graph(
            id="attention_weights_graph",
            figure=attention_weights,
            style={"width": "100%", "height": "100%"},
        )
        attention_matrices_graph = dcc.Graph(
            id="attention_matrices_graph",
            figure=attention_matrices,
            style={"width": "100%", "height": "100%"},
        )
        return attention_weights_graph, attention_matrices_graph
