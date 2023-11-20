from . import Input, Output, State, html
from . import dash_table
from . import PreventUpdate
from . import callback_context, get_input_trigger, columns_map

from . import Datasets
from . import FileHandler
from . import DatasetConfig


def register_load_callbacks(app):
    analysis_folder = f'/Users/ay227/Desktop/Final-Year/Datasets'
    fh = FileHandler(analysis_folder)
    dataset_obj = DatasetConfig()

    # @app.callback(
    #
    #     Output('download-link', 'href'),
    #     [
    #         Input('download', 'n_clicks'),
    #     ]
    # )
    # def download(n_clicks):
    #     if n_clicks > 0:
    #         fig = app.layout['test'].figure
    #         import os
    #         filename = f'/Users/ay227/Desktop/test.pdf'
    #         os.makedirs(os.path.dirname(filename), exist_ok=True)
    #         fig.write_image(filename)
    #         return filename
    #     else:
    #         raise PreventUpdate

    @app.callback(
        [
            Output('dataset_prompt', 'children'),
            Output('model_name', 'options'),
            Output('split', 'options'),
        ],
        [
            Input('dataset_name', 'value')
        ]
    )
    def populate_dataset_options(dataset_name):
        # if dataset_name is not None or len(dataset_name)<1:
        if dataset_name is not None:
            dataset = Datasets[dataset_name]
            model_names = list(dataset['model_name'].keys())
            splits = list(dataset['splits'].keys())
            return html.Div(f'The Chosen Dataset: {dataset_name}', style={'color': 'green'}), model_names, splits
        else:
            return html.Div('Please Choose a Dataset', style={'color': 'red'}), [], []

    @app.callback(
        Output('load_dataset_prompt', 'children'),
        [
            Input('load_dataset', 'n_clicks'),
            State('dataset_name', 'value'),
            State('model_name', 'value'),
            State('split', 'value')
        ]
    )
    def load_data(n_clicks, dataset_name, model_name, split):
        # if dataset_name is not None or len(dataset_name)<1:
        if n_clicks > 0:
            if model_name is None:
                return html.Div('Please Choose the Model', style={'color': 'red'})
            elif split is None:
                return html.Div('Please Choose the Split', style={'color': 'red'})
            else:
                dataset = Datasets[dataset_name]
                model_path = dataset['model_name'][model_name]
                split = dataset['splits'][split]
                dataset_obj.load_data(fh, dataset_name, model_name, model_path, split)
                return html.Div(f'Dataset Loaded : {round(dataset_obj.dataset_end_time, 4)} Minutes',
                                style={'color': 'green'})
        else:
            return html.Div('Please Load the Dataset', style={'color': 'red'})

    @app.callback(
        Output('initialized_dataset_prompt', 'children'),
        [
            Input('initialize_model', 'n_clicks'),
            State('dataset_name', 'value'),
            State('model_name', 'value'),
            State('split', 'value')
        ]
    )
    def initialize_model(n_clicks, dataset_name, model_name, split):
        # if dataset_name is not None or len(dataset_name)<1:
        if n_clicks > 0:
            if dataset_obj.created:
                dataset_obj.initialize_model()
                return html.Div(f'Model Initialized: {round(dataset_obj.initialize_end_time, 4)} Minute',
                                style={'color': 'green'})
            else:
                dataset_obj.create_model()
                return html.Div(f'Model Created: {round(dataset_obj.create_end_time, 4)} Minute',
                                style={'color': 'green'})
        else:
            return html.Div('Please Initialize the Model', style={'color': 'red'})

    @app.callback(
        Output("dataset_table_container", "children"),
        [
            Input("view_dataset", "n_clicks"),
            Input("hide_dataset", "n_clicks")
        ]

    )
    def create_dataset_table(view, hide):
        ctx = callback_context
        input_trigger = get_input_trigger(ctx)
        if dataset_obj.loaded:
            if input_trigger == "view_dataset":
                table_data = dataset_obj.analysis_df.copy()
                table_data = table_data.rename(columns=columns_map)

                return dash_table.DataTable(
                    id='dataset_table',
                    columns=[
                        {'name': i, 'id': i, 'deletable': True} for i in table_data.columns
                        # omit the id column
                        if i != 'id'
                    ],
                    style_header={'text-align': 'center', 'background-color': '#555555',
                                  'color': 'white'},
                    data=table_data.head(100).to_dict('records'),
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
            else:
                return []
        else:
            raise PreventUpdate

    return dataset_obj

