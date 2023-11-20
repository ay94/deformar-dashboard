import dash

from . import Input, Output, State, html, dcc, callback_context
from . import PreventUpdate
from . import identify_mistakes, color_tokens, color_map, min_max, \
    get_input_trigger, default_view
from . import px, go, head_view, model_view, torch, cosine_similarity, \
    tqdm, np, pd
from . import AttentionSimilarity, Datasets

from . import hover_data, train_hover_data


def register_instance_callbacks(app, dataset_obj):
    @app.callback(
        [
            Output('initialize_instance_tab', 'children'),
            Output('initialize_instance', 'children'),
            Output('compare_status', 'children'),
            Output('similarity_status', 'children'),
        ],
        [
            Input("Tabs", "value"),
        ]

    )
    def initialize_tab(tab):
        if tab == 'instance':
            if dataset_obj.initialized:
                div = html.Div('Model Initialized', style={'color': 'green'})
            else:
                div = html.Div('Please Initialize the Model', style={'color': 'red'})
            return div, div, div, div
        else:
            raise PreventUpdate

    @app.callback(
        [
            Output("instance_label_map", "children"),
            Output("instance_sentence", "children"),
            Output("instance_truth", "children"),
            Output("instance_pred", "children"),
            Output("instance_mistakes", "children"),
            Output('instance_tokens', 'options'),
            Output('instance_scatter', 'figure'),
        ],
        [
            Input("visualize_instance", "n_clicks"),
            State("error_instances", "value"),
            State('performance_scatter_mode', 'value'),
        ]
    )
    def visualize_instance(n_clicks, example_id, scatter_mode):
        if n_clicks == 0:
            raise PreventUpdate
        elif n_clicks > 0:
            try:
                instance_df = dataset_obj.analysis_df[dataset_obj.analysis_df['Sentence Id'] == str(example_id)].copy()
                example_words = dataset_obj.corpus[dataset_obj.split][int(example_id)][1]
                example_labels = dataset_obj.corpus[dataset_obj.split][int(example_id)][2]
                label_map = dataset_obj.corpus['labels']
            except:
                instance_df = dataset_obj.analysis_df[dataset_obj.analysis_df['Sentence Id'] == '0'].copy()
                example_words = dataset_obj.corpus[dataset_obj.split][0][1]
                example_labels = dataset_obj.corpus[dataset_obj.split][0][2]
                label_map = dataset_obj.corpus['labels']

            tokens = list(instance_df['Anchor Token'])
            option_df = instance_df[['Token Selector', 'Words']]
            token_ids = []
            for _, row in option_df.iterrows():
                token_ids.append({'label': row['Token Selector'], 'value': row['Words']})
            labels = list(instance_df['Ground Truth'])
            preds = list(instance_df['Prediction'])
            mistakes = identify_mistakes(tokens, labels, preds)

            label_color_map, colored_words, colored_truth_text, colored_pred_text = color_tokens(example_words,
                                                                                                 example_labels,
                                                                                                 label_map, tokens,
                                                                                                 labels, preds)

            instance_fig = px.scatter(
                instance_df, x="X Coordinate", y="Y Coordinate",
                color='Ground Truth',
                symbol='Class Agreement',
                color_discrete_map=color_map,
                template='ggplot2',
                hover_data=hover_data)
            if 'group' in scatter_mode:
                instance_fig.update_layout(scattermode="group")

            x_range, y_range = min_max(instance_df, 0.5)
            instance_fig.update_xaxes(
                range=x_range
            )
            instance_fig.update_yaxes(
                range=y_range
            )

        return label_color_map, colored_words, colored_truth_text, colored_pred_text, mistakes, token_ids, instance_fig

    @app.callback(
        [
            Output('pre_attention_view', 'srcDoc'),
            Output('fin_attention_view', 'srcDoc'),
            Output('instance_training_impact', 'figure'),
        ],
        [
            Input("visualize_training_impact", "n_clicks"),
            Input("generate_errors", "n_clicks"),
            Input("visualize_instance", "n_clicks"),
            State("impact_instances", "value"),
            State("attention_view", "value"),
        ]
    )
    def visualize_training_impact(n_clicks, generate_errors, visualize_instance, example_id, chosen_view):
        if dataset_obj.initialized:
            ctx = callback_context
            input_trigger = get_input_trigger(ctx)
            if input_trigger == 'generate_errors' or input_trigger == 'visualize_instance':
                return "", "", go.Figure()
            if n_clicks == 0:
                raise PreventUpdate
            elif n_clicks > 0:

                split_data = dataset_obj.instanceLevel.test_dataset
                try:
                    example = split_data.__getitem__(int(example_id))
                except:
                    example = split_data.__getitem__(0)

                input_ids = example['input_ids'][example['input_ids'] != 0]
                attention_mask = example['attention_mask'][example['input_ids'] != 0]
                token_type_ids = example['token_type_ids'][example['input_ids'] != 0]
                tokens = dataset_obj.instanceLevel.tokenizer.convert_ids_to_tokens(input_ids)
                inputs = {'input_ids': input_ids[None, :],
                          'attention_mask': attention_mask[None, :],
                          'token_type_ids': token_type_ids[None, :]
                          }

                pre_output = dataset_obj.pretrained_bert(**inputs)
                fin_output = dataset_obj.finetuned.bert(**inputs)
                pre_attention = pre_output[-1]
                fin_attention = fin_output[-1]
                view = default_view(chosen_view)
                if view == 'head':
                    pre_vis = head_view(pre_attention, tokens, html_action='return')
                    fin_vis = head_view(fin_attention, tokens, html_action='return')
                elif view == 'model':
                    pre_vis = model_view(pre_attention, tokens, html_action='return')
                    fin_vis = model_view(fin_attention, tokens, html_action='return')

                attention_impact = AttentionSimilarity(torch.device("cpu"),
                                                       dataset_obj.pretrained_bert,
                                                       dataset_obj.finetuned.bert,
                                                       dataset_obj.instanceLevel.tokenizer,
                                                       dataset_obj.instanceLevel.preprocessor)

                scores = attention_impact.compute_similarity(
                    dataset_obj.corpus[dataset_obj.split][int(example_id)][1]
                )

                training_impact_fig = px.imshow(scores,
                                                labels=dict(x="Heads", y="Layers", color="Similarity Score"),
                                                )

            return pre_vis.data, fin_vis.data, training_impact_fig
        return "", "", go.Figure()

    @app.callback(
        Output('example_split', 'options'),
        Input("visualize_instance", "n_clicks"),
    )
    def populate_split(n_clicks):
        if dataset_obj.loaded:
            if n_clicks > 0:
                if dataset_obj.split == 'train':
                    columns = [
                        {'label': 'Train', 'value': 'train'},
                    ]
                else:
                    columns = [
                        {'label': 'Train', 'value': 'train'},
                        {'label': dataset_obj.split.capitalize(), 'value': dataset_obj.split},
                    ]
                return columns
            else:
                return []
        else:
            raise PreventUpdate

    @app.callback(
        [
            Output('choose_example', 'options'),
            Output('example_tokens_similarity', 'options'),
            Output('example_token_comparison', 'options'),
            Output('examples_scatter', 'figure'),
            Output('examples_status', 'children'),
        ],
        [
            Input("load_token_data", "n_clicks"),
            State("instance_tokens", "options"),
            State("instance_tokens", "value"),
            State("example_split", "value"),
            State('performance_scatter_mode', 'value'),
        ]
    )
    def load_token_data(n_clicks, tokens, value, split, scatter_mode):
        if n_clicks == 0:
            raise PreventUpdate
        elif n_clicks > 0:
            if split == 'train':
                data = dataset_obj.light_train_df.copy()
                data['Anchor Token'] = data['Token Selector'].apply(lambda x: x.split('@#')[0])
                data['Sentence Id'] = data['Token Selector'].apply(lambda x: x.split('@#')[1])
            else:
                data = dataset_obj.analysis_df.copy()
            try:
                selected_label = [option['label'] for option in tokens if option['value'] == value][0]
                token = selected_label.split('@#')[0]
            except:
                raise PreventUpdate
            sen_ids = data[data['Anchor Token'].isin([token])]['Sentence Id']
            output_data = data[data['Sentence Id'].isin(sen_ids.values)]
            token_ids = data[data['Anchor Token'].isin([token])]['Token Selector']
            if len(sen_ids) >= 1 and len(token_ids) >= 1:

                x_range, y_range = min_max(output_data, 0.2)
                examples_fig = px.scatter(
                    output_data, x="X Coordinate", y="Y Coordinate",
                    color='Ground Truth',
                    symbol='Class Agreement',
                    color_discrete_map=color_map,
                    template='ggplot2',
                    hover_data=hover_data)
                if 'group' in scatter_mode:
                    examples_fig.update_layout(scattermode="group")
                examples_fig.update_xaxes(
                    range=x_range
                )
                examples_fig.update_yaxes(
                    range=y_range
                )

                return sen_ids, token_ids, token_ids, examples_fig, html.Div('Examples Loaded',
                                                                             style={'color': 'green'})
            else:
                return [], [], [], go.Figure(), html.Div('No Data Loaded', style={'color': 'red'})

    @app.callback(
        [
            Output("example_label_map", "children"),
            Output("example_sentence", "children"),
            Output("example_truth", "children"),
            Output("example_pred", "children"),
            Output("example_mistakes", "children"),
            Output('example_scatter', 'figure'),
        ],
        [
            Input("visualize_token_example", "n_clicks"),
            State("choose_example", "value"),
            State("example_split", "value"),
            State("instance_tokens", "value"),
            State('performance_scatter_mode', 'value'),
        ]
    )
    def visualize_example(n_clicks, example_id, split, instance_token, scatter_mode):
        if n_clicks == 0:
            raise PreventUpdate
        elif n_clicks > 0:
            label_map = dataset_obj.corpus['labels']
            if split == 'train':
                data = dataset_obj.light_train_df.copy()
                examples = dataset_obj.corpus[split]
            else:
                data = dataset_obj.analysis_df.copy()
                examples = dataset_obj.corpus[split]
            try:
                instance_df = data[data['Sentence Id'] == example_id].copy()
                example_words = examples[int(example_id)][1]
                example_labels = examples[int(example_id)][2]
            except:
                instance_df = data[data['Sentence Id'] == 0].copy()
                example_words = examples[0][1]
                example_labels = examples[0][2]

            tokens = list(instance_df['Anchor Token'])
            labels = list(instance_df['Ground Truth'])
            preds = list(instance_df['Prediction'])
            mistakes = identify_mistakes(tokens, labels, preds)

            label_color_map, colored_words, colored_truth_text, colored_pred_text = color_tokens(example_words,
                                                                                                 example_labels,
                                                                                                 label_map, tokens,
                                                                                                 labels, preds,
                                                                                                 instance_token)

            x_range, y_range = min_max(instance_df, 0.2)
            example_fig = px.scatter(
                instance_df, x="X Coordinate", y="Y Coordinate",
                color='Ground Truth',
                symbol='Class Agreement',
                color_discrete_map=color_map,
                template='ggplot2',
                hover_data=train_hover_data)
            if 'group' in scatter_mode:
                example_fig.update_layout(scattermode="group")
            example_fig.update_xaxes(
                range=x_range
            )
            example_fig.update_yaxes(
                range=y_range
            )

        return label_color_map, colored_words, colored_truth_text, colored_pred_text, mistakes, example_fig

    @app.callback(
        Output('example_similarity_matrix', 'figure'),
        [
            Input("compute_example_similarity_matrix", "n_clicks"),
            Input("example_tokens_similarity", "options"),
            State("example_tokens_similarity", "value"),
            State("example_split", "value"),
        ]

    )
    def compute_example_similarity_matrix(n_clicks, tokens, current_tokens, split):
        if dataset_obj.initialized and tokens is not None:
            if n_clicks == 0 or len(tokens) < 2:
                raise PreventUpdate
            elif n_clicks > 0:
                if current_tokens is None or len(current_tokens) == 0:
                    chosen_tokens = tokens
                else:
                    chosen_tokens = current_tokens
                if split == 'train':
                    output_data = dataset_obj.light_train_df.copy()
                    split_data = dataset_obj.instanceLevel.train_dataset
                else:
                    output_data = dataset_obj.analysis_df.copy()
                    split_data = dataset_obj.instanceLevel.test_dataset

                sen_ids = output_data[output_data['Token Selector'].isin(chosen_tokens)][
                    ['Sentence Id', 'Token Selector']].values
                examples = []
                with torch.no_grad():
                    for sen_id, token in tqdm(sen_ids):
                        example = split_data.__getitem__(int(sen_id))
                        inputs = {'input_ids': example['input_ids'][example['input_ids'] != 0][None, :],
                                  'attention_mask': example['attention_mask'][example['input_ids'] != 0][None, :],
                                  'token_type_ids': example['token_type_ids'][example['input_ids'] != 0][None, :]
                                  }
                        bert_output = dataset_obj.finetuned.bert(**inputs)

                        examples.append(bert_output.last_hidden_state[0][int(token.split('@#')[2])].tolist())
                similarities = cosine_similarity(np.array(examples))
                matrix = pd.DataFrame(similarities, columns=chosen_tokens, index=chosen_tokens)
                similarity_matrix_fig = px.imshow(matrix)
        else:
            raise PreventUpdate

        return similarity_matrix_fig

    @app.callback(

        Output('example_compare_similarity', 'figure'),

        [
            Input("compare_example_similarity", "n_clicks"),
            Input("example_token_comparison", "options"),
            State('example_token_comparison', 'value'),
            State("instance_tokens", "options"),
            State("instance_tokens", "value"),
            State("example_split", "value"),
        ]
    )
    def compare_example_similarity(n_clicks, tokens, current_tokens, example_token, value, split):
        if dataset_obj.initialized and tokens is not None:
            if n_clicks == 0 or len(tokens) < 1:
                raise PreventUpdate
            elif n_clicks > 0:
                if current_tokens is None or len(current_tokens) == 0:
                    chosen_tokens = tokens
                else:
                    chosen_tokens = current_tokens
                selected_label = [option['label'] for option in example_token if option['value'] == value][0]
                locator = selected_label.split('@#')
                compare = dataset_obj.instanceLevel.test_dataset
                compare_example = compare.__getitem__(int(locator[1]))
                compare_inputs = {'input_ids': compare_example['input_ids'][compare_example['input_ids'] != 0][None, :],
                                  'attention_mask': compare_example['attention_mask'][
                                                        compare_example['input_ids'] != 0][
                                                    None,
                                                    :],
                                  'token_type_ids': compare_example['token_type_ids'][
                                                        compare_example['input_ids'] != 0][
                                                    None,
                                                    :]
                                  }
                compare_output = dataset_obj.finetuned.bert(**compare_inputs)
                compare_hidden_state = compare_output.last_hidden_state[0][int(locator[2])].detach().numpy()
                if split == 'train':
                    output_data = dataset_obj.light_train_df.copy()
                    split_data = dataset_obj.instanceLevel.train_dataset
                else:
                    output_data = dataset_obj.analysis_df.copy()
                    split_data = dataset_obj.instanceLevel.test_dataset

                sen_ids = output_data[output_data['Token Selector'].isin(chosen_tokens)][
                    ['Sentence Id', 'Token Selector']].values
                examples = []
                with torch.no_grad():
                    for sen_id, token in tqdm(sen_ids):
                        example = split_data.__getitem__(int(sen_id))
                        inputs = {'input_ids': example['input_ids'][example['input_ids'] != 0][None, :],
                                  'attention_mask': example['attention_mask'][example['input_ids'] != 0][None, :],
                                  'token_type_ids': example['token_type_ids'][example['input_ids'] != 0][None, :]
                                  }
                        bert_output = dataset_obj.finetuned.bert(**inputs)

                        examples.append(bert_output.last_hidden_state[0][int(token.split('@#')[2])].detach().numpy())
                similarity_scores = cosine_similarity(compare_hidden_state.reshape(1, -1), examples)

                similarity_data = pd.DataFrame(
                    {'tokens': chosen_tokens, 'similarity': similarity_scores.flatten().tolist()})
                example_comparison_bar = px.bar(similarity_data, x="tokens", y="similarity")

            return example_comparison_bar
        else:
            return go.Figure()
