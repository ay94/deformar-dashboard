from . import Input, Output, State, html
from . import dash_table
from . import PreventUpdate
from . import callback_context, get_input_trigger

from . import Datasets
from . import FileHandler
from . import DatasetConfig

columns_map = {
    'global_id': 'Global Id', 'token_id': 'Token Id', 'word_id': 'Word Id',
    'sen_id': 'Sentence Id', 'token_ids': 'Token Selector', 'label_ids': 'Label Id',
    'first_tokens_freq': 'First Token Frequency', 'first_tokens_consistency': 'First Token Consistency',
    'first_tokens_inconsistency': 'First Token Inconsistency', 'words': 'Words', 'wordpieces': 'Word Pieces',
    'tokens': 'Tokens', 'first_tokens': 'First Token', 'truth': 'Ground Truth', 'pred': 'Prediction',
    'agreement': 'Class Agreement', 'losses': 'Loss', 'tokenization_rate': 'Tokenization Rate',
    'token_entropy': 'Token Entropy', 'word_entropy': 'Word Entropy', 'tr_entity': 'Entity Truth',
    'pr_entity': 'Entity Prediction', 'error_type': 'Error Type', 'prediction_entropy': 'Prediction Entropy',
    'confidences': 'Confidence', 'variability': 'Variability', 'O': 'O Confidence', 'B-PERS': 'B-PERS Confidence',
    'I-PERS': 'I-PERS Confidence', 'B-ORG': 'B-ORG Confidence', 'I-ORG': 'I-ORG Confidence',
    'B-LOC': 'B-LOC Confidence',
    'I-LOC': 'I-LOC Confidence', 'B-MISC': 'B-MISC Confidence', 'I-MISC': 'I-MISC Confidence', '3_clusters': 'K=3',
    '4_clusters': 'K=4', '9_clusters': 'K=9', 'truth_token_score': 'Truth Silhouette Score',
    'pred_token_score': 'Prediction Silhouette Score',
    'x': 'X Coordinate', 'y': 'Y Coordinate', 'pre_x': 'Pretrained X Coordinate', 'pre_y': 'Pretrained Y Coordinate'
}

characteristics_columns = [
    'global_id', 'token_id', 'word_id', 'sen_id', 'token_ids', 'label_ids',
    'words', 'wordpieces', 'tokens', 'first_tokens', 'truth', 'pred', 'agreement',
    'tr_entity', 'pr_entity', 'error_type',
    'first_tokens_freq', 'first_tokens_consistency', 'first_tokens_inconsistency',
    'tokenization_rate',
    'token_entropy', 'word_entropy',
    'losses',   'prediction_entropy', 'confidences', 'variability',
    'truth_token_score', 'pred_token_score',
    'O', 'B-PERS', 'I-PERS', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC',
    '3_clusters', '4_clusters', '9_clusters'
]

decision_columns = [
    'global_id', 'token_id', 'word_id', 'sen_id', 'token_ids', 'label_ids',
    'first_tokens_freq', 'first_tokens_consistency', 'first_tokens_inconsistency',
    'words', 'wordpieces', 'tokens', 'first_tokens', 'truth', 'pred', 'agreement',
    'losses', 'x', 'y', 'tokenization_rate', 'token_entropy', 'word_entropy', 'tr_entity',
    'pr_entity', 'error_type', 'prediction_entropy', 'confidences', 'variability',
    'O', 'B-PERS', 'I-PERS', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC',
    '3_clusters', '4_clusters', '9_clusters', 'truth_token_score', 'pred_token_score', 'pre_x', 'pre_y'
]

performance_columns = [
    'global_id', 'token_id', 'word_id', 'sen_id', 'token_ids', 'token_ids', 'label_ids',
    'first_tokens_freq', 'first_tokens_consistency', 'first_tokens_inconsistency',
    'words', 'wordpieces', 'tokens', 'first_tokens', 'truth', 'pred', 'agreement',
    'losses', 'x', 'y', 'tokenization_rate', 'token_entropy', 'word_entropy', 'tr_entity',
    'pr_entity', 'error_type', 'prediction_entropy', 'confidences', 'variability',
    'O', 'B-PERS', 'I-PERS', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC',
    '3_clusters', '4_clusters', '9_clusters', 'truth_token_score', 'pred_token_score', 'pre_x', 'pre_y'
]

instance_columns = [
    'global_id', 'token_id', 'word_id', 'sen_id', 'token_ids', 'token_ids', 'label_ids',
    'first_tokens_freq', 'first_tokens_consistency', 'first_tokens_inconsistency',
    'words', 'wordpieces', 'tokens', 'first_tokens', 'truth', 'pred', 'agreement',
    'losses', 'x', 'y', 'tokenization_rate', 'token_entropy', 'word_entropy', 'tr_entity',
    'pr_entity', 'error_type', 'prediction_entropy', 'confidences', 'variability',
    'O', 'B-PERS', 'I-PERS', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC',
    '3_clusters', '4_clusters', '9_clusters', 'truth_token_score', 'pred_token_score', 'pre_x', 'pre_y'
]


def register_dataset_callbacks(app):
    analysis_folder = f'/Users/ay227/Desktop/Final-Year/Datasets'
    fh = FileHandler(analysis_folder)
    dataset_obj = DatasetConfig()

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

                return dash_table.DataTable(
                    id='dataset_table',
                    columns=[
                        {'name': i, 'id': i, 'deletable': True} for i in dataset_obj.analysis_df.columns
                        # omit the id column
                        if i != 'id'
                    ],
                    style_header={'text-align': 'center', 'background-color': '#555555',
                                  'color': 'white'},
                    data=dataset_obj.analysis_df.head(100).to_dict('records'),
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
    # if dataset_obj.loaded:
    #     return dataset_obj
    # else:
    #     return None
