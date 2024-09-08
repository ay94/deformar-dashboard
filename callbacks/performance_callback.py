# from . import (Input, Output, PreventUpdate, State, callback_context,
#                color_map, compute_confusion, cosine_similarity,
#                create_confusion_table, create_error_bars,
#                create_token_confusion, dash, dash_table, dcc, default_color,
#                default_coordinates, default_entity, extract_column,
#                get_input_trigger, get_value, go, hover_data, html, min_max, np,
#                pd, px, torch, tqdm)


def register_error_callbacks(app, dataset_obj):
    pass
    # @app.callback(
    #     [
    #         Output("initialize_performance_tab", "children"),
    #         Output("error_similarity_status", "children"),
    #     ],
    #     [
    #         Input("Tabs", "value"),
    #         Input("initialize_error_tab", "n_clicks"),
    #     ],
    # )
    # def initialize_tab(tab, n_clicks):
    #     if n_clicks > 0 and dataset_obj.loaded:
    #         initialization_div = html.Div("Tab Initialized", style={"color": "green"})
    #     else:
    #         initialization_div = None
    #     if tab == "performance":
    #         if dataset_obj.initialized:
    #             div = html.Div("Model Initialized", style={"color": "green"})
    #         else:
    #             div = html.Div("Please Initialize the Model", style={"color": "red"})
    #         return initialization_div, div
    #     else:
    #         raise PreventUpdate

    # @app.callback(
    #     [
    #         Output("error_datatable", "data"),
    #         Output("error_datatable", "columns"),
    #         Output("filter_column", "value"),
    #         Output("filter_value", "value"),
    #     ],
    #     [
    #         Input("initialize_error_tab", "n_clicks"),
    #         Input("filter_error_table", "n_clicks"),
    #         Input("reset_error_table", "n_clicks"),
    #         State("filter_column", "value"),
    #         State("filter_value", "value"),
    #     ],
    #     prevent_initial_call=True,
    # )
    # def populate_error_table(
    #     initialize_table, filter_table, reset_table, filter_column, filter_value
    # ):
    #     ctx = callback_context
    #     input_trigger = get_input_trigger(ctx)
    #     if input_trigger == "initialize_error_tab" and dataset_obj.loaded:
    #         columns = [
    #             {"name": i, "id": i, "deletable": True}
    #             for i in dataset_obj.light_df.columns
    #             # omit the id column
    #             if i != "id"
    #         ]
    #         output_data = dataset_obj.light_df.to_dict("records")

    #         return [output_data, columns, dash.no_update, dash.no_update]
    #     else:

    #         if input_trigger == "reset_error_table":
    #             output_data = dataset_obj.light_df.copy().to_dict("records")
    #             filter_column = None
    #             filter_value = ""
    #         elif input_trigger == "filter_error_table":

    #             data = dataset_obj.analysis_df.copy()
    #             filter_col = extract_column(filter_column)
    #             try:
    #                 if len(filter_value.split()) > 1:
    #                     filter_values = filter_value.split()
    #                 else:
    #                     filter_values = [filter_value]

    #                 output_data = data[data[filter_col].isin(filter_values)].to_dict(
    #                     "records"
    #                 )
    #             except:
    #                 raise PreventUpdate
    #         try:
    #             return [output_data, dash.no_update, filter_column, filter_value]
    #         except:
    #             raise PreventUpdate

    # @app.callback(
    #     [
    #         Output("error_context_color", "options"),
    #         Output("error_x", "options"),
    #         Output("error_y", "options"),
    #         Output("error_color", "options"),
    #         Output("filter_column", "options"),
    #     ],
    #     Input("initialize_error_tab", "n_clicks"),
    #     prevent_initial_call=True,
    # )
    # def add_error_columns(n_clicks):
    #     if n_clicks > 0 and dataset_obj.loaded:
    #         return [
    #             dataset_obj.analysis_df.columns[10:],
    #             dataset_obj.analysis_df.columns[10:],
    #             dataset_obj.analysis_df.columns[10:],
    #             dataset_obj.analysis_df.columns[10:],
    #             dataset_obj.analysis_df.columns[6:],
    #         ]
    #     else:
    #         return [[], [], [], [], []]

    # @app.callback(
    #     Output("entity", "options"),
    #     Input("performance_metric", "value"),
    # )
    # def add_entity_options(metric):
    #     if metric == "errors" and dataset_obj.loaded:
    #         return dataset_obj.entity_prediction["Entity"].unique()
    #     else:
    #         raise PreventUpdate

    # @app.callback(
    #     [
    #         Output("performance_container_1", "children"),
    #         Output("performance_container_2", "children"),
    #     ],
    #     [
    #         Input("compute_metric", "n_clicks"),
    #         State("performance_metric", "value"),
    #         State("entity", "value"),
    #     ],
    # )
    # def compute_performance_metric(n_clicks, metric, entity):

    #     if n_clicks > 0 and dataset_obj.loaded:

    #         if metric == "report":
    #             performance_container_1 = dash_table.DataTable(
    #                 data=dataset_obj.seq_report.to_dict("records"),
    #                 columns=[
    #                     {"name": i, "id": i} for i in dataset_obj.seq_report.columns
    #                 ],
    #                 style_data={"text-align": "center"},
    #                 style_cell={"padding": "10px"},
    #                 style_header={
    #                     "text-align": "center",
    #                     "background-color": "#555555",
    #                     "color": "white",
    #                 },
    #             )
    #             performance_container_2 = dash_table.DataTable(
    #                 data=dataset_obj.skl_report.to_dict("records"),
    #                 columns=[
    #                     {"name": i, "id": i} for i in dataset_obj.skl_report.columns
    #                 ],
    #                 style_data={"text-align": "center"},
    #                 style_cell={"padding": "10px"},
    #                 style_header={
    #                     "text-align": "center",
    #                     "background-color": "#555555",
    #                     "color": "white",
    #                 },
    #             )
    #         elif metric == "confusion":
    #             performance_container_1 = dcc.Graph(
    #                 id="entity_confusion_matrix",
    #                 figure=px.imshow(
    #                     compute_confusion(dataset_obj.confusion_data),
    #                     labels=dict(
    #                         x="Prediction", y="Truth", color="Number of Instances"
    #                     ),
    #                     text_auto=True,
    #                     template="ggplot2",
    #                 ).update_layout(title="Entity Confusion Matrix"),
    #             )

    #             token_data = create_token_confusion(dataset_obj.analysis_df)
    #             performance_container_2 = dcc.Graph(
    #                 id="token_confusion_matrix",
    #                 figure=px.imshow(
    #                     compute_confusion(token_data),
    #                     labels=dict(
    #                         x="Prediction", y="Truth", color="Number of Mistakes"
    #                     ),
    #                     text_auto=True,
    #                     template="ggplot2",
    #                 ).update_layout(title="Token Error Confusion Matrix"),
    #             )
    #         else:
    #             chosen_entity = default_entity(entity)
    #             confusion_table, entity_confusion_table = create_error_bars(
    #                 dataset_obj.analysis_df,
    #                 dataset_obj.entity_prediction,
    #                 chosen_entity,
    #             )

    #             performance_container_1 = dcc.Graph(
    #                 id="entity_confusion_bar",
    #                 figure=px.bar(
    #                     entity_confusion_table,
    #                     orientation="h",
    #                     color_discrete_map=color_map,
    #                 ).update_layout(
    #                     title="Entity Error Barchart",
    #                     xaxis_title="Prediction",
    #                     yaxis_title="Truth",
    #                 ),
    #             )
    #             performance_container_2 = dcc.Graph(
    #                 id="token_confusion_bar",
    #                 figure=px.bar(
    #                     confusion_table, orientation="h", color_discrete_map=color_map
    #                 ).update_layout(
    #                     title="Token Error Barchart",
    #                     xaxis_title="Prediction",
    #                     yaxis_title="Truth",
    #                 ),
    #             )
    #     else:
    #         raise PreventUpdate

    #         # Display the confusion matrix
    #     return [performance_container_1, performance_container_2]

    # @app.callback(
    #     [
    #         Output("error_context_scatter", "figure"),
    #         Output("error_scatter", "figure"),
    #     ],
    #     [
    #         Input("generate_errors", "n_clicks"),
    #         Input("filter_error_table", "n_clicks"),
    #         Input("reset_error_table", "n_clicks"),
    #         Input("error_datatable", "derived_virtual_row_ids"),
    #         Input("error_context_color", "value"),
    #         Input("error_color", "value"),
    #         State("error_x", "value"),
    #         State("error_y", "value"),
    #         State("filter_column", "value"),
    #         State("filter_value", "value"),
    #         State("draw_text", "value"),
    #         State("performance_scatter_mode", "value"),
    #     ],
    # )
    # def update_error_scatter(
    #     generate_errors,
    #     filter_table,
    #     reset_table,
    #     row_ids,
    #     error_context_color,
    #     error_color,
    #     error_x,
    #     error_y,
    #     filter_column,
    #     filter_value,
    #     checked,
    #     scatter_mode,
    # ):
    #     if generate_errors == 0:
    #         raise PreventUpdate
    #     elif generate_errors > 0:
    #         color, symbol = default_color(error_color)
    #         x, y = default_coordinates(error_x, error_y)
    #         context_color, context_symbol = default_color(error_context_color)
    #         output_data = dataset_obj.analysis_df.copy()
    #         if row_ids is None:
    #             dff = dataset_obj.analysis_df.copy()
    #             # pandas Series works enough like a list for this to be OK
    #         else:
    #             if None not in row_ids:
    #                 dff = dataset_obj.analysis_df.loc[row_ids].copy()
    #             # if the button was clicked once then make sure you generate dff this way
    #             if filter_table > 0 and filter_value:

    #                 filter_col = extract_column(filter_column)
    #                 if len(filter_value.split()) > 1:
    #                     filter_values = filter_value.split()
    #                 else:
    #                     filter_values = [filter_value]
    #                 try:
    #                     dff = dataset_obj.analysis_df[
    #                         dataset_obj.analysis_df[filter_col].isin(filter_values)
    #                     ]
    #                 except:
    #                     raise PreventUpdate
    #         if len(output_data) == len(dff):
    #             dff = dff[dff["Class Agreement"] == False]

    #         output_data.loc[dff["Global Id"], context_color] = "Selected"

    #         ctx = callback_context
    #         input_trigger = get_input_trigger(ctx)
    #         if input_trigger == "error_example_ids":
    #             raise PreventUpdate
    #         error_context_fig = px.scatter(
    #             output_data,
    #             x=x,
    #             y=y,
    #             color=context_color,
    #             symbol=context_symbol,
    #             color_discrete_map=color_map,
    #             template="ggplot2",
    #             hover_data=hover_data,
    #         )
    #         if "group" in scatter_mode:
    #             error_context_fig.update_layout(scattermode="group")

    #         error_fig = px.scatter(
    #             dff,
    #             x=x,
    #             y=y,
    #             color=color,
    #             symbol=symbol,
    #             color_discrete_map=color_map,
    #             template="ggplot2",
    #             hover_data=hover_data,
    #         )
    #         if "group" in scatter_mode:
    #             error_fig.update_layout(scattermode="group")
    #         x_range, y_range = min_max(dff, 0.5)

    #         error_fig.update_xaxes(range=x_range)
    #         error_fig.update_yaxes(range=y_range)
    #         subset = dff.sample(n=min(100, len(dff)), random_state=42)

    #         if "draw_text" in checked:
    #             for i in range(subset.shape[0]):
    #                 error_fig.add_annotation(
    #                     x=subset.iloc[i]["X Coordinate"],
    #                     y=subset.iloc[i]["Y Coordinate"],
    #                     text=subset.iloc[i]["Anchor Token"],
    #                     showarrow=True,
    #                     font=dict(size=15, color="black"),
    #                 )

    #     return [error_context_fig, error_fig]

    # @app.callback(
    #     [
    #         Output("error_similarity_tokens", "options"),
    #         Output("error_tokens", "options"),
    #         Output("error_example_ids", "children"),
    #         Output("compute_error_similarity_matrix", "n_clicks"),
    #     ],
    #     [
    #         Input("error_context_scatter", "selectedData"),
    #         State("error_x", "value"),
    #         State("error_y", "value"),
    #     ],
    # )
    # def selected_context_errors(selected_errors, x, y):
    #     if selected_errors is None or len(selected_errors["points"]) == 0:
    #         return [], [], [], 0
    #     else:
    #         x, y = default_coordinates(x, y)
    #         selected_point = pd.DataFrame(selected_errors["points"])[["x", "y"]]
    #         selected_point = selected_point.rename(
    #             columns={"x": "X Coordinate", "y": "Y Coordinate"}
    #         )

    #         selected_rows = dataset_obj.analysis_df.merge(
    #             selected_point,
    #             on=[x, y],
    #         )
    #     return [
    #         selected_rows["Token Selector"].values,
    #         selected_rows["Anchor Token"].values,
    #         selected_rows[["Sentence Id", "Token Selector"]].to_json(orient="split"),
    #         0,
    #     ]

    # @app.callback(
    #     Output("error_token_ambiguity", "figure"),
    #     [
    #         Input("error_tokens", "value"),
    #         Input("error_context_scatter", "selectedData"),
    #     ],
    # )
    # def error_token_ambiguity(error_tokens, error_data):
    #     if error_data is None:
    #         return go.Figure()
    #     else:
    #         if error_tokens is None or len(error_tokens) < 1:
    #             return go.Figure()
    #         else:
    #             return dataset_obj.token_ambiguity.visualize_ambiguity(error_tokens)

    # @app.callback(
    #     Output("error_similarity_matrix", "figure"),
    #     [
    #         Input("compute_error_similarity_matrix", "n_clicks"),
    #         Input("error_similarity_tokens", "options"),
    #         State("error_similarity_tokens", "value"),
    #     ],
    # )
    # def compute_similarity_matrix(n_clicks, tokens, chosen_tokens):
    #     if n_clicks == 0 and tokens is None and chosen_tokens is None:
    #         raise PreventUpdate
    #     elif n_clicks > 0:
    #         if chosen_tokens is None or len(chosen_tokens) == 0:
    #             chosen_tokens = tokens
    #         elif tokens is None or len(tokens) == 0:
    #             chosen_tokens = tokens
    #         else:
    #             chosen_tokens = chosen_tokens
    #         output_data = dataset_obj.analysis_df.copy()
    #         split_data = dataset_obj.instanceLevel.test_dataset
    #         sen_ids = output_data[output_data["Token Selector"].isin(chosen_tokens)][
    #             ["Sentence Id", "Token Selector"]
    #         ].values
    #         examples = []
    #         with torch.no_grad():
    #             for sen_id, token in tqdm(sen_ids):
    #                 example = split_data.__getitem__(int(sen_id))
    #                 inputs = {
    #                     "input_ids": example["input_ids"][example["input_ids"] != 0][
    #                         None, :
    #                     ],
    #                     "attention_mask": example["attention_mask"][
    #                         example["input_ids"] != 0
    #                     ][None, :],
    #                     "token_type_ids": example["token_type_ids"][
    #                         example["input_ids"] != 0
    #                     ][None, :],
    #                 }
    #                 bert_output = dataset_obj.finetuned.bert(**inputs)

    #                 examples.append(
    #                     bert_output.last_hidden_state[0][
    #                         int(token.split("@#")[2])
    #                     ].tolist()
    #                 )
    #         try:
    #             similarities = cosine_similarity(np.array(examples))
    #             matrix = pd.DataFrame(
    #                 similarities, columns=chosen_tokens, index=chosen_tokens
    #             )
    #             similarity_matrix = px.imshow(matrix)
    #         except:
    #             raise PreventUpdate
    #     else:
    #         return go.Figure()

    #     return similarity_matrix

    # @app.callback(
    #     [
    #         Output("error_instances", "options"),
    #         Output("impact_instances", "options"),
    #     ],
    #     [
    #         Input("error_example_ids", "children"),
    #         Input("error_datatable", "derived_virtual_row_ids"),
    #         Input("filter_error_table", "n_clicks"),
    #         Input("Tabs", "value"),
    #         State("filter_column", "value"),
    #         State("filter_value", "value"),
    #     ],
    # )
    # def update_error_instances(
    #     saved_points, row_ids, filter_table, tab, filter_columns, filter_value
    # ):
    #     if dataset_obj.loaded:
    #         ctx = callback_context
    #         input_trigger = get_input_trigger(ctx)
    #         if input_trigger == "filter_error_table":
    #             data = dataset_obj.analysis_df.copy()
    #             filter_col = extract_column(filter_columns)
    #             try:
    #                 if len(filter_value.split()) > 1:
    #                     filter_values = filter_value.split()
    #                 else:
    #                     filter_values = [filter_value]

    #                 output_data = data[data[filter_col].isin(filter_values)]
    #                 example_ids = list(output_data["Sentence Id"].unique())
    #             except:
    #                 raise PreventUpdate

    #         elif saved_points is not None and len(saved_points) != 0:
    #             selected = pd.read_json(saved_points, orient="split")
    #             example_ids = list(selected["Sentence Id"].unique())
    #         elif row_ids is not None and len(row_ids) > 0:
    #             if None not in row_ids:
    #                 dff = dataset_obj.analysis_df.loc[row_ids].copy()
    #                 example_ids = list(dff["Sentence Id"].unique())
    #             else:
    #                 dff = dataset_obj.analysis_df.copy()
    #                 example_ids = list(dff["Sentence Id"].unique())

    #         else:
    #             dff = dataset_obj.analysis_df.copy()
    #             example_ids = list(dff["Sentence Id"].unique())
    #     else:
    #         return [[], []]

    #     return [example_ids, example_ids]
