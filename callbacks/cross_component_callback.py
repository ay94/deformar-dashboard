import plotly.graph_objs as go
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State, MATCH
from dash.exceptions import PreventUpdate

from managers.tabs.cross_component_tab_managers import DataTabManager, ModelTabManager, EvaluationTabManager

from managers.tabs.registries import make_data_registry, make_model_registry, make_evaluation_registry  # or inline the dicts above



def register_callbacks(app, variants_data, config_manager):
    data_mgr  = DataTabManager(variants_data, config_manager)
    model_mgr = ModelTabManager(variants_data, config_manager)
    eval_mgr = EvaluationTabManager(variants_data, config_manager)

    REGISTRIES = {
        "data":  make_data_registry(data_mgr),
        "model": make_model_registry(model_mgr),
        "evaluation": make_evaluation_registry(eval_mgr),
    }


    @app.callback(
        Output("loaded_tab_data", "children"), Input("language_selector", "value")
    )
    def update_output(variant):
        tab_data = data_mgr.get_tab_data(variant)
        if tab_data:
            return html.Div(
                f"Tab data loaded and Variant {variant} is chosen.",
                className="prompt-message",
            )
        else:
            return html.Div("Tab data is not available", className="prompt-message")
        
    @app.callback(
        [
            Output({"type": "figure-dd", "component": MATCH}, "options"),
            Output({"type": "figure-dd", "component": MATCH}, "value"),
        ],
        Input({"type": "analysis-dd", "component": MATCH}, "value"),
        State({"type": "analysis-dd", "component": MATCH}, "id"),
    )
    def update_figures(analysis_id, analysis_dd_id):
        if not analysis_id:
            return [], None

        component_id = analysis_dd_id.get("component")

        comp_cfg = next(
            (c for c in config_manager.cross_component.get("components", [])
            if c.get("id") == component_id),
            None,
        )
        if not comp_cfg:
            return [], None

        ana = next((a for a in comp_cfg.get("analyses", [])
                    if a.get("id") == analysis_id), None)
        figs = ana.get("figures", []) if ana else []
        options = [{"label": f["label"], "value": f["id"]} for f in figs]

        return options, None

    
 
    @app.callback(
        Output({"type": "figure-canvas", "component": MATCH}, "children"),
        Input({"type": "render-btn", "component": MATCH}, "n_clicks"),
        State({"type": "analysis-dd", "component": MATCH}, "id"),
        State({"type": "figure-dd", "component": MATCH}, "value"),
        State("language_selector", "value"),
        prevent_initial_call=True,
    )
    def render_canvas(n_clicks, analysis_dd_id, figure_id, variant):
        if not n_clicks or not analysis_dd_id or not figure_id:
            raise PreventUpdate
        component_id = analysis_dd_id.get("component")
        registry = REGISTRIES.get(component_id, {})
        handler = registry.get(figure_id)
        if not handler:
            return html.Div("No renderer for this figure.", className="text-muted")
        return handler(variant)
    

    # def render_dataset_stats_cb(n_clicks, analysis_id, figure_id, selected_variant):
    #     if not n_clicks or not analysis_id or not figure_id:
    #         raise PreventUpdate
    #     if figure_id != "dataset_stats":
    #         return html.Div("No renderer for the selected figure.", className="text-muted")

    #     table = tab_manager.generate_dataset_stats(selected_variant)   # TRANSFORM (helper via manager)
    #     return table






    # @app.callback(
    #     Output("summary_container", "children"),
    #     [
    #         Input("generate_summary", "n_clicks"),
    #         State("statistical_columns", "value"),
    #         State("variant_selector", "value"),
    #     ],
    # )
    # def generate_summary_statistics(n_clicks, statistical_columns, variant):
    #     if n_clicks == 0:
    #         raise PreventUpdate

    #     # Generate the summary statistics
    #     summary_table = tab_manager.generate_summary_statistics(
    #         variant, statistical_columns
    #     )

    #     if summary_table is None:
    #         return html.Div(
    #             "Please select a type of results and click 'View Results' to view data.",
    #             className="prompt-message",
    #         )
    #     return summary_table

    # @app.callback(
    #     Output("distribution_container", "children"),
    #     [
    #         Input("plot_distribution", "n_clicks"),
    #         State("distribution_column", "value"),
    #         State("categorical_column", "value"),
    #         State("variant_selector", "value"),
    #     ],
    # )
    # def plot_distribution(n_clicks, distribution_column, categorical_column, variant):
    #     if n_clicks == 0:
    #         raise PreventUpdate

    #     # Generate the figure using the TabManager's method
    #     figure = tab_manager.generate_distribution_or_violin(
    #         variant=variant,
    #         distribution_column=distribution_column,
    #         categorical_column=categorical_column,
    #     )

    #     # Check if a valid figure was returned
    #     if isinstance(figure, go.Figure):
    #         return dcc.Loading(
    #             id="distribution_graph_loader",
    #             type="default",
    #             children=[dcc.Graph(figure=figure)],
    #             style={
    #                 "width": "100%",
    #                 "height": "100%",
    #             },  # Ensure the loader covers only the graph size
    #         )

    #     return html.Div(
    #         "Please select a  type and click 'Plot Distribution' to view plot'.",
    #         className="prompt-message",
    #     )

    # @app.callback(
    #     Output("results_output_container", "children"),
    #     [
    #         Input("view_results_type", "n_clicks"),
    #         State("results_type", "value"),
    #         State("variant_selector", "value"),
    #     ],
    # )
    # def view_results(n_clicks, results_type, variant):
    #     if n_clicks == 0:
    #         raise PreventUpdate

    #     # Get the results data using the tab manager
    #     results_data = tab_manager.get_results_data(variant, results_type)

    #     if results_data is None:
    #         return html.Div(
    #             "Please select a type of results and click 'View Results' to view data.",
    #             className="prompt-message",
    #         )

    #     # Use the CustomDataTable class to create the table
    #     return results_data

    # @app.callback(
    #     Output("custom_analysis_graph_container", "children"),
    #     [Input("plot_custom_analysis", "n_clicks")],
    #     [State("custom_analysis_type", "value"), State("variant_selector", "value")],
    # )
    # def custom_analysis_callback(n_clicks, custom_distribution_type, variant):
    #     if n_clicks == 0:
    #         raise PreventUpdate
    #     # find the index of last underscore and take everything except it. 
    #     corpus_name = variant[:variant.rfind('_')]

    #     analysis = tab_manager.perform_custom_analysis(
    #         custom_distribution_type, variant, corpus_name
    #     )
    #     # Check if a valid figure was returned
    #     if isinstance(analysis, go.Figure):
    #         return dcc.Graph(figure=analysis)

    #     if analysis is None:
    #         return html.Div(
    #             "Please select a  type and click 'Plot Custom Distribution' to view data'.",
    #             className="prompt-message",
    #         )

    #     return analysis

    # @app.callback(
    #     [
    #         Output("correlation_matrix_graph", "figure"),
    #         Output("correlation_scatter_graph", "figure"),
    #         Output("correlation_matrix_graph", "style"),
    #         Output("correlation_scatter_graph", "style"),
    #         Output("correlation_prompt", "children"),  # Message output container
    #     ],
    #     [
    #         Input("calculate_correlation", "n_clicks"),
    #         Input("correlation_matrix_graph", "clickData"),  # Reference the Graph ID
    #     ],
    #     [
    #         State("correlation_coefficient", "value"), 
    #         State("correlation_categorical_column", "value"),
    #         State("variant_selector", "value")
    #     ],
    # )
    # def plot_correlation(n_clicks, clickData, correlation_method, categorical_column, variant):
    #     # Default to no update
    #     correlation_fig = no_update
    #     scatter_fig = no_update
    #     correlation_graph_style = {"width": "49%", "display": "none"}
    #     scatter_graph_style = {"width": "49%", "display": "none"}
    #     message_prompt = html.Div(
    #             "Please select a correlation coefficient and variant before calculating.",
    #             className="prompt-message",
    #         )
        
    #      # Check if button click is valid
    #     if n_clicks is None or n_clicks == 0:
    #         # No clicks yet, do not trigger the update
    #         return correlation_fig, scatter_fig, correlation_graph_style, scatter_graph_style, message_prompt

    #     # Validate user inputs (correlation_method and variant)
    #     if correlation_method is None:
            
    #         return correlation_fig, scatter_fig, correlation_graph_style, scatter_graph_style, message_prompt

    #     # Check if button click is valid
    #     if n_clicks is not None and n_clicks > 0:
    #         # Calculate the initial correlation and scatter plot figures
    #         correlation_fig, scatter_fig = tab_manager.calculate_correlation(
    #             variant=variant,
    #             correlation_method=correlation_method,
    #             categorical_column=categorical_column
    #         )
    #         # Set styles to display the graphs
    #         correlation_graph_style = {"width": "49%", "display": "inline-block"}
    #         scatter_graph_style = {"width": "49%", "display": "inline-block"}
    #         message_prompt = ""  # Default to no message

    #     # Check if a matrix cell was clicked
    #     if clickData is not None:
    #         x_column = clickData["points"][0]["x"]
    #         y_column = clickData["points"][0]["y"]
    #         # Update only the scatter plot based on the clicked cell
    #         correlation_fig, scatter_fig = tab_manager.calculate_correlation(
    #             variant=variant,
    #             correlation_method=correlation_method,
    #             categorical_column=categorical_column,
    #             x_column=x_column,
    #             y_column=y_column,
    #         )
    #         # Ensure styles remain set to display the graphs
    #         correlation_graph_style = {"width": "49%", "display": "inline-block"}
    #         scatter_graph_style = {"width": "49%", "display": "inline-block"}

    #     # Return figures and styles
    #     return (
    #         correlation_fig,
    #         scatter_fig,
    #         correlation_graph_style,
    #         scatter_graph_style,
    #         message_prompt
    #     )
