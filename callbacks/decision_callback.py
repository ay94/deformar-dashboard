from . import Input, Output, State, html
from . import dash_table
from . import PreventUpdate
from . import callback_context, get_input_trigger, default_color, \
    color_map, default_coordinates, create_confusion_table, defualt_centroid
from . import px, pd, go


def register_decision_callbacks(app, dataset_obj):
    @app.callback(
        [
            Output("selector_datatable", "columns"),
            Output("selector_datatable", "data")
        ],
        Input("initialize_decision_tab", "n_clicks"),

        prevent_initial_call=True
    )
    def populate_decision_selector_table(n_clicks):
        if n_clicks > 0 and dataset_obj.loaded:
            columns = [
                {'name': i, 'id': i, 'deletable': True} for i in dataset_obj.light_df.columns
                # omit the id column
                if i != 'id'
            ]
            data = dataset_obj.light_df.to_dict('records')
            return [columns, data]

        else:
            raise PreventUpdate

    @app.callback(
        Output("selector_scatter", "figure"),
        [
            Input("initialize_decision_tab", "n_clicks"),
            Input('selector_datatable', 'derived_virtual_row_ids'),
            Input('selector_datatable', 'selected_row_ids'),
        ]
    )
    def update_selector_scatter(n_clicks, row_ids, selected_row_ids):
        if n_clicks > 0 and dataset_obj.loaded:
            selected_id_set = set(selected_row_ids or [])
            output_data = dataset_obj.light_df.copy()
            if row_ids is None:
                dff = output_data

            else:
                if None not in row_ids:
                    dff = output_data.loc[row_ids]

            filter_mask = output_data['global_id'].isin(dff['global_id'])
            selection_mask = output_data['global_id'].isin(selected_id_set)

            # Use the boolean mask and if-else statement to assign values to the 'colors' column
            output_data.loc[filter_mask, 'colors'] = 'Filtered'
            output_data.loc[~filter_mask, 'colors'] = 'Not Filtered'
            output_data.loc[selection_mask, 'colors'] = 'Selected'

            fig = px.scatter(
                output_data, x="x", y="y",
                color="colors",
                symbol="error_type",
                hover_data=['token_ids', 'agreement', 'confidences', 'variability', 'truth', 'pred'],
                template='ggplot2')
            fig.update_layout(scattermode="group")
            return fig

        else:
            raise PreventUpdate

    @app.callback(
        [
            Output("decision_columns", "options"),
            Output("measure_columns", "options"),
            Output("measure_x", "options"),
            Output("measure_y", "options"),
            Output("centroid_columns", "options"),
            Output("centroid_cluster", "options")
        ],
        Input("initialize_decision_tab", "n_clicks"),
        prevent_initial_call=True
    )
    def add_columns(n_clicks):
        if n_clicks > 0 and dataset_obj.loaded:
            return [
                dataset_obj.analysis_df.columns[6:],
                dataset_obj.analysis_df.columns[6:],
                dataset_obj.analysis_df.columns[6:],
                dataset_obj.analysis_df.columns[6:],
                dataset_obj.centroid_df.columns[5:],
                dataset_obj.centroid_df['centroid'].unique()
            ]
        else:
            return [
                [], [], [], [], [], [],
            ]

    @app.callback(
        Output("decision_scatter", "figure"),
        Input('generate_decision', 'n_clicks'),
        Input("save_measure_points", "children"),
        State('decision_columns', 'value'),
        prevent_initial_call=True
    )
    def create_decision_plot(n_clicks, saved_points, columns):
        if n_clicks == 0:
            raise PreventUpdate
        elif n_clicks > 0:
            color, symbol = default_color(columns)
            if saved_points is None:
                data = dataset_obj.analysis_df
            else:
                data = dataset_obj.analysis_df.copy()
                selected = pd.read_json(saved_points, orient="split")
                data.loc[selected['global_id'], color] = 'Selected'
            try:
                decision_plot = px.scatter(
                    data, x="x", y="y",
                    color=color,
                    symbol=symbol,
                    hover_data=['token_ids', 'agreement', 'confidences', 'variability', 'truth', 'pred'],
                    color_discrete_sequence=px.colors.qualitative.Light24_r,
                    color_discrete_map=color_map,
                    template='ggplot2')
                decision_plot.update_layout(scattermode="group")
            except:
                decision_plot = px.scatter(
                    data, x="x", y="y",
                    color=color,
                    hover_data=['token_ids', 'agreement', 'confidences', 'variability', 'truth', 'pred'],
                    color_discrete_sequence=px.colors.qualitative.Light24_r,
                    color_discrete_map=color_map,
                    template='ggplot2')
                decision_plot.update_layout(scattermode="group")

        return decision_plot

    @app.callback(
        Output("measure_scatter", "figure"),
        Input('generate_measure', 'n_clicks'),
        Input('include_ignored', 'value'),
        Input("save_decision_points", "children"),
        State('measure_columns', 'value'),
        State('measure_x', 'value'),
        State('measure_y', 'value'))
    def create_measure_plot(n_clicks, include_ignored, saved_points, measure_color, x, y):
        if n_clicks == 0:
            raise PreventUpdate
        elif n_clicks > 0:
            x, y = default_coordinates(x, y)
            color, symbol = default_color(measure_color)

            if 'checked' in include_ignored:
                measure_data = dataset_obj.analysis_df.copy()
            else:
                measure_data = dataset_obj.analysis_df[
                    ~dataset_obj.analysis_df['first_tokens'].isin(['IGNORED', '[CLS]', '[SEP]'])].copy()
            if saved_points is None:
                data = measure_data
            else:
                data = measure_data.copy()
                selected = pd.read_json(saved_points, orient="split")
                data.loc[selected['global_id'], color] = 'Selected'

            try:
                measure_plot = px.scatter(
                    data, x=x, y=y,
                    color=color,
                    symbol=symbol,
                    hover_data=['token_ids', 'agreement', 'confidences', 'variability', 'prediction_entropy',
                                'token_entropy', 'truth', 'pred'],
                    color_discrete_sequence=px.colors.qualitative.Light24_r,
                    color_discrete_map=color_map,
                    template='ggplot2')
                measure_plot.update_layout(scattermode="group")
            except:
                measure_plot = px.scatter(
                    data, x=x, y=y,
                    color=color,
                    hover_data=['token_ids', 'agreement', 'confidences', 'variability', 'prediction_entropy',
                                'token_entropy', 'truth', 'pred'],
                    color_discrete_sequence=px.colors.qualitative.Light24_r,
                    color_discrete_map=color_map,
                    template='ggplot2')
                measure_plot.update_layout(scattermode="group")

        return measure_plot

    @app.callback(
        Output("save_decision_points", "children"),
        [
            Input("decision_scatter", "selectedData"),
            Input('include_ignored', 'value'),
            Input('reset_decision', 'n_clicks')
        ]
    )
    def save_decision_selection(selectedData, include_ignored, n_clicks):
        ctx = callback_context
        input_trigger = get_input_trigger(ctx)
        if input_trigger == 'reset':
            return None
        if selectedData is None or len(selectedData['points']) == 0:
            raise PreventUpdate
        else:
            selected_point = pd.DataFrame(selectedData["points"])[["x", "y"]]

            selected_rows = dataset_obj.analysis_df.merge(
                selected_point,
                on=["x", "y"],
            )
        if 'checked' in include_ignored:
            measure_data = selected_rows.copy()
        else:
            measure_data = selected_rows[~selected_rows['first_tokens'].isin(['IGNORED', '[CLS]', '[SEP]'])].copy()

        return measure_data.to_json(orient="split")

    @app.callback(
        Output("save_measure_points", "children"),
        [
            Input("measure_scatter", "selectedData"),
            Input('reset_measure', 'n_clicks'),
            State('measure_x', 'value'),
            State('measure_y', 'value')
        ]
    )
    def save_measure_selection(selectedData, n_clicks, x, y):
        ctx = callback_context
        input_trigger = get_input_trigger(ctx)
        x, y = default_coordinates(x, y)
        if input_trigger == 'reset_measure':
            return None
        if selectedData is None or len(selectedData['points']) == 0:
            raise PreventUpdate
        else:
            selected_point = pd.DataFrame(selectedData["points"])[["x", "y"]]
            selected_point = selected_point.rename(columns={'x': x, 'y': y})

            selected_rows = dataset_obj.analysis_df.merge(
                selected_point,
                on=[x, y],
            )
        return selected_rows.to_json(orient="split")

    @app.callback(
        Output("centroid_scatter", "figure"),
        [
            Input("generate_centroid", "n_clicks"),
            State("centroid_columns", "value"),
            State("centroid_cluster", "value"),
        ]
    )
    def create_centroid_plot(n_clicks, centroid_color, centroid_cluster):
        if n_clicks == 0:
            raise PreventUpdate
        elif n_clicks > 0:
            color, symbol = default_color(centroid_color)
            centroid_cluster = defualt_centroid(centroid_cluster)
            data = dataset_obj.centroid_df[dataset_obj.centroid_df['centroid'] == centroid_cluster]

            try:
                centroid_scatter = px.scatter(
                    data, x='x', y='y',
                    color=color,
                    symbol=symbol,
                    hover_data=data.columns,
                    color_discrete_sequence=px.colors.qualitative.Light24_r,
                    color_discrete_map=color_map,
                    template='ggplot2')
                centroid_scatter.update_layout(scattermode="group")
            except:
                fig = px.scatter(
                    data, x='x', y='y',
                    color=color,
                    hover_data=data.columns,
                    color_discrete_sequence=px.colors.qualitative.Light24_r,
                    color_discrete_map=color_map,
                    template='ggplot2')
                centroid_scatter.update_layout(scattermode="group")
        return centroid_scatter

    @app.callback(
        [
            Output("selection_tag_proportion", "figure"),
            Output("selection_datatable", "children")
        ],

        [
            Input("save_decision_points", "children"),
            Input("save_measure_points", "children")
        ]
    )
    def update_selection_datatable(decision_points, measure_points):
        if decision_points is not None:
            points = pd.read_json(decision_points, orient="split")
        elif measure_points is not None:
            points = pd.read_json(measure_points, orient="split")
        else:
            return [None, go.Figure()]

        selection_datatable = dash_table.DataTable(
            id='selecting',
            columns=[
                {'name': i, 'id': i, 'deletable': True} for i in points.columns
                # omit the id column
                if i != 'id'
            ],
            style_header={'text-align': 'center', 'background-color': '#555555',
                          'color': 'white'},
            data=points.to_dict('records'),
            editable=True,
            filter_action="native",
            sort_action="native",
            sort_mode='multi',
            column_selectable="single",
            row_selectable='multi',
            row_deletable=True,
            selected_rows=[],
            page_action='native',
            page_current=0,
            page_size=10,
        ),
        points.groupby(['truth', 'pred'])['global_id'].count().reset_index()
        confusion_table = create_confusion_table(points)
        selection_tag_proportion = px.bar(confusion_table, orientation='h')
        selection_tag_proportion.update_layout(
            xaxis_title='Pred',
            yaxis_title='Truth'
        )

        return [selection_tag_proportion, selection_datatable]

    @app.callback(
        Output("selection_token_ambiguity", "figure"),
        [
            Input("save_decision_points", "children"),
            Input("save_measure_points", "children"),
        ],
    )
    def token_ambiguity(scatter_points, measure_points):

        if scatter_points is not None:
            points = pd.read_json(scatter_points, orient="split")
        elif measure_points is not None:
            points = pd.read_json(measure_points, orient="split")
        if len(points) < 1:
            raise PreventUpdate

        return dataset_obj.token_ambiguity.visualize_ambiguity(list(points['first_tokens'].unique()))

    @app.callback(
        Output("impact_plot", "figure"),
        [
            Input("show_impact", "n_clicks"),
            State("impact_view", "value"),
        ]


    )
    def show_impact(n_clicks, view):
        if n_clicks == 0:
            raise PreventUpdate
        elif n_clicks > 0:
            if view == 'activations':
                impact_plot = dataset_obj.activations.update_layout(
                                title="Training Impact View"
                            )
            else:
                impact_plot = dataset_obj.weights

        return impact_plot.update_layout(
                                title="Training Impact View"
                            )
