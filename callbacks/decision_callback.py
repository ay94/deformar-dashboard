import json
import pandas as pd
import dash
import plotly.graph_objs as go
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from layouts.managers.layout_managers import (get_input_trigger,
                                              process_selection, render_basic_table_with_font)
from managers.tabs.decision_tab_mangers import DecisionTabManager
DISPLAY_COLUMNS = [
        "Global Id", "Sentence Ids", "Words", "Tokens", "Token Selector Id",  
        "Token Ambiguity", "Word Ambiguity", "Consistency Ratio",
        "Inconsistency Ratio", "Tokenization Rate", "Token Confidence",
        "Loss Values", "Prediction Uncertainty", "True Silhouette", "Pred Silhouette",
    ]

def register_callbacks(app, variants_data):
    tab_manager = DecisionTabManager(variants_data)
    
   
    @app.callback(
        [   
            Output("training_graph", "children"),
            Output("training_sentences", "options")
        ],
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
            ), ['No Selection']
        fig, sentence_ids = tab_manager.get_training_data(variant)
        if fig is None:
            return html.Div(
                "Please extract training data or click the button.",
                className="prompt-message",
            ), ['No Selection']
        return dcc.Graph(figure=fig), sentence_ids
    
    
    @app.callback(
        [
            Output("training_sentence", "children"),
            Output("training_truth", "children"),
        ],
        [
            Input("variant_selector", "value"),
            Input("training_sentences", "value"),
        ]
    )
    def update_instance_display(variant, instance_id):
        if not variant or instance_id is None:
            raise PreventUpdate
        sentence_colored, truth_colored = tab_manager.generate_training_output(variant, instance_id)
        return sentence_colored, truth_colored
    
    

    @app.callback(
        [
            Output("training_entity_true_iob", "children"),
            Output("training_entity_true_iob2", "children"),
        ],
        [
            Input("variant_selector", "value"),
            Input("training_sentences", "value"),
        ],
    )
    def update_entity_annotations(variant, instance_id):
        if not variant or instance_id is None:
            raise PreventUpdate

        true_iob, words_iob = tab_manager.get_training_entity_level_annotations_non_strict(variant, instance_id)
        rendered_true_iob = tab_manager.render_training_entity_tags(true_iob, words_iob)
    

        true_iob2, words_iob2 = tab_manager.get_training_entity_level_annotations_non_strict(variant, instance_id)
        rendered_true_iob2 = tab_manager.render_training_entity_tags(true_iob2, words_iob2)
       

        return (
            rendered_true_iob,
            rendered_true_iob2,
        )
    
    @app.callback(
        Output("filter_value_dropdown", "options"),
        Input("filter_column_dropdown", "value"),
        State("variant_selector", "value"),  # or whatever triggers data context
    )
    def update_filter_value_options(selected_column, variant):
        if not selected_column or not variant:
            return []

        tab_data = tab_manager.get_tab_data(variant)
        if not tab_data or tab_data.analysis_data.empty:
            return []

        df = tab_data.analysis_data  # or whatever your main df is
        unique_values = df[selected_column].dropna().unique()

        # Convert to dropdown options
        return [{"label": str(val), "value": val} for val in sorted(unique_values)]
    
    @app.callback(
        [
            Output("filtered_data_table", "data"),
            Output("filtered_data_table", "columns"),
            Output("filter_column_dropdown", "value"),
            Output("filter_value_dropdown", "value"),
        ],
        [
            Input("variant_selector", "value"),
            Input("filter_table_button", "n_clicks"),
            Input("reset_filter_button", "n_clicks"),
        ],
        [
            State("filter_column_dropdown", "value"),
            State("filter_value_dropdown", "value"),
        ],
    )
    def update_filtered_table(variant_value, filter_clicks, reset_clicks, selected_column, selected_value):
        ctx = dash.callback_context
        trigger_id = get_input_trigger(ctx)
        # Apply filter
        if trigger_id == "filter_table_button":
            if not selected_column or not selected_value:
                raise dash.exceptions.PreventUpdate

            df = tab_manager.get_filtered_analysis_data(variant_value, selected_column, selected_value)
            if df is None or df.empty:
                return [], [], no_update, no_update

            return (
                df.to_dict("records"),
                [{"name": col, "id": col} for col in df.columns],
                no_update,
                no_update,
            )

        # Reset: full table, clear dropdowns
        elif trigger_id == "reset_filter_button":
            df = tab_manager.get_filtered_analysis_data(variant_value)
            if df is None or df.empty:
                return [], [], None, None

            return (
                df.to_dict("records"),
                [{"name": col, "id": col} for col in df.columns],
                None,
                None,
            )

        df = tab_manager.get_filtered_analysis_data(variant_value)
        return (
            df.to_dict("records"),
            [{"name": col, "id": col} for col in df.columns],
            no_update,
            no_update,
        )

    @app.callback(
        Output("filter_state", "data"),
        Input("filtered_data_table", "derived_virtual_data"),
        prevent_initial_call=True
    )
    def detect_manual_table_filtering(filtered_data):
        ctx = dash.callback_context
        trigger = get_input_trigger(ctx)
        # Only treat it as a manual filter if it was triggered by the table
        if trigger != "filtered_data_table":
            raise PreventUpdate
        if filtered_data is None or len(filtered_data) == 0:
            return {"filtered": False}
        
        # If all rows have valid IDs, we assume it’s filtered
        is_filtered = all("id" in row and row["id"] is not None for row in filtered_data)
        return {"filtered": is_filtered}



    @app.callback(
        [
            Output("correlation_heatmap", "figure"),
            Output("correlation_heatmap", "style"),
        ],
        [
            Input("view_decision_boundary", "n_clicks"),
            Input("filtered_data_table", "derived_virtual_data"),  # 👈 new input
        ],
        [
            State("correlation_columns", "value"),
            State("decision_correlation_coefficient", "value"),
            State("variant_selector", "value"),
        ],
    )
    def generate_matrix(n_clicks, filtered_rows, correlation_columns, correlation_method, variant):
        # Default to no update
        matrix_fig = no_update
        matrix_style = {"width": "100%", "height": "500px", "display": "none"}

        # Check if button click is valid
        if n_clicks is not None and n_clicks > 0:
            # Calculate the initial correlation and scatter plot figures
            correlation_method = correlation_method if correlation_method else "Pearson"
            selected_values = correlation_columns if correlation_columns else None
            if filtered_rows:
                # Convert the filtered rows back into a DataFrame
                
                df = pd.DataFrame(filtered_rows)
                matrix_fig = tab_manager.generate_matrix_from_df(
                    df=df,
                    correlation_method=correlation_method,
                    selected_columns=selected_values,
                )
            else:
                matrix_fig = tab_manager.generate_matrix(
                variant=variant,
                correlation_method=correlation_method,
                selected_columns=selected_values,
            )

            matrix_style["display"] = "inline-block"
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
            State("model_type", "value"),
            State("decision_columns", "value"),
        ],
    )
    def generate_decision_plot(
        n_clicks, measureSelection, variant, model_type, color_columns
    ):

        fig = no_update
        style = {"width": "100%", "height": "500px", "display": "none"}

        if n_clicks is not None and n_clicks > 0:

            model_type = model_type if model_type else "Fine Tuned Model"
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
                decision_type=model_type,
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
            Input("filtered_data_table", "derived_virtual_data"),  # new input
            Input("filter_state", "data"),
        ],
        [
            State("variant_selector", "value"),
            State("measure_columns", "value"),
            State("model_type", "value"),
        ],
    )
    def generate_measure_plot(
        n_clicks, clickData, decisionSelection, filtered_rows, filter_state, variant, color_columns, model_type
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
        print('I am in the selecgtion measrue plot trigger', len(selection_ids))
        

        # Use filtered table data if available
        is_filtered = filtered_rows and any("id" in row and row["id"] is not None for row in filtered_rows)

       
        if filter_state.get("filtered"):
            print('Yes I am filtered')
            filtered_row_ids = [row["id"] for row in filtered_rows if "id" in row]
            fig = tab_manager.generate_measure_plot_from_ids(
                ids=filtered_row_ids,
                variant=variant,
                model_type=model_type,
                x_column=x_column,
                y_column=y_column,
                color_column=color_column,
                symbol_column=symbol_column,
                selection_ids=selection_ids,
            )
        else:
            # Fallback to full variant
            print('I am very full')
            fig = tab_manager.generate_measure_plot(
                variant=variant,
                model_type=model_type,
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
        Output("cluster_results_container", "children"),
        [
            Input("variant_selector", "value"),  # Correctly using 'data' property
        ]
    )
    def generate_kmeans_results(variant):
        kmeans_results = tab_manager.generate_kmeans_results(variant)
        
        if kmeans_results is None or kmeans_results.empty:
            # No data selected for either case
            return html.Div(
                "No data KMeans results.",
                className="prompt-message",
            )
        
        
        return render_basic_table_with_font(kmeans_results)
    
        
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
            Output("selection_summary_container", "children"),
            Output("selection_numeric_summary_container", "children"),
        ],
        [
            Input("measure_store", "data"),  # Correctly using 'data' property
            Input("decision_store", "data"),
            Input("selection_summary_column", "value"),  # Correctly using 'data' property
        ],
        [
            State("variant_selector", "value"),
        ]
    )
    def generate_selection_summary(measureSelection, decisionSelection, column, variant):
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
            msg = html.Div(
                "No data selected from measure or decision plots.",
                className="prompt-message",
            )
            return [msg, msg]  # ✅ Correct: returns 2 outputs as a list
            
        summary = tab_manager.generate_selection_summary(variant=variant, category=column, selection_ids=selection_ids)
            
        if summary is None or column is None:
            # No data selected for either case
            msg = html.Div(
                    "No summary available please choose category.",
                    className="prompt-message",
                )
            return [msg, msg]  # ✅ Correct: returns 2 outputs as a list
        # return [
        #     render_basic_table_with_font(summary["categorical_summary"]),
        #     render_basic_table_with_font(summary["numeric_summary"]),
        #     ]
        return [
            html.Div([
                html.H5("Categorical Summary"),
                render_basic_table_with_font(summary["categorical_summary"]),
            ]),
            html.Div([
                html.Hr(),
                html.H5("Metric Summary"),
                render_basic_table_with_font(summary["numeric_summary"]),
            ], style={"marginTop": "30px"}),
        ]

   
    
    
    @app.callback(
        Output("selection_tag_container", "children"),
        [
            Input("measure_store", "data"),  # Correctly using 'data' property
            Input("decision_store", "data"),
            Input("selection_tag_column", "value"),  # Correctly using 'data' property
        ],
        [
            State("variant_selector", "value"),
        ],
    )
    def generate_selection_output(measureSelection, decisionSelection, column, variant):
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
            variant=variant, column=column, selection_ids=selection_ids
        )
        if isinstance(fig, go.Figure):
            return dcc.Graph(figure=fig, style={"width": "100%", "height": "500px"})


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
