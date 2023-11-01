from . import Input, Output, State, html
from . import dash_table
from . import PreventUpdate
from . import callback_context, get_input_trigger
from . import px, go, make_subplots, np, pd, gaussian_kde, dcc
from . import Datasets
from . import FileHandler
from . import DatasetConfig

# columns_map = {
#     'global_id': 'Global Id', 'token_id': 'Token Id', 'word_id': 'Word Id',
#     'sen_id': 'Sentence Id', 'token_ids': 'Token Selector', 'label_ids': 'Label Id',
#     'first_tokens_freq': 'First Token Frequency', 'first_tokens_consistency': 'First Token Consistency',
#     'first_tokens_inconsistency': 'First Token Inconsistency', 'words': 'Words', 'wordpieces': 'Word Pieces',
#     'tokens': 'Tokens', 'first_tokens': 'First Token', 'truth': 'Ground Truth', 'pred': 'Prediction',
#     'agreement': 'Class Agreement', 'losses': 'Loss', 'tokenization_rate': 'Tokenization Rate',
#     'token_entropy': 'Token Entropy', 'word_entropy': 'Word Entropy', 'tr_entity': 'Entity Truth',
#     'pr_entity': 'Entity Prediction', 'error_type': 'Error Type', 'prediction_entropy': 'Prediction Entropy',
#     'confidences': 'Confidence', 'variability': 'Variability', 'O': 'O Confidence', 'B-PERS': 'B-PERS Confidence',
#     'I-PERS': 'I-PERS Confidence', 'B-ORG': 'B-ORG Confidence', 'I-ORG': 'I-ORG Confidence',
#     'B-LOC': 'B-LOC Confidence',
#     'I-LOC': 'I-LOC Confidence', 'B-MISC': 'B-MISC Confidence', 'I-MISC': 'I-MISC Confidence', '3_clusters': 'K=3',
#     '4_clusters': 'K=4', '9_clusters': 'K=9', 'truth_token_score': 'Truth Silhouette Score',
#     'pred_token_score': 'Prediction Silhouette Score',
#     'x': 'X Coordinate', 'y': 'Y Coordinate', 'pre_x': 'Pretrained X Coordinate', 'pre_y': 'Pretrained Y Coordinate'
# }

# summary_map = {
#     'first_tokens_freq': 'First Token Frequency', 'first_tokens_consistency': 'First Token Consistency',
#     'first_tokens_inconsistency': 'First Token Inconsistency',
#     'agreement': 'Class Agreement', 'losses': 'Loss', 'tokenization_rate': 'Tokenization Rate',
#     'token_entropy': 'Token Entropy', 'word_entropy': 'Word Entropy', 'tr_entity': 'Entity Truth',
#     'prediction_entropy': 'Prediction Entropy',
#     'confidences': 'Confidence', 'variability': 'Variability',
#     'O': 'O Confidence', 'B-PERS': 'B-PERS Confidence', 'I-PERS': 'I-PERS Confidence', 'B-ORG': 'B-ORG Confidence',
#     'I-ORG': 'I-ORG Confidence',
#     'B-LOC': 'B-LOC Confidence', 'I-LOC': 'I-LOC Confidence', 'B-MISC': 'B-MISC Confidence',
#     'I-MISC': 'I-MISC Confidence',
#     '3_clusters': 'K=3', '4_clusters': 'K=4', '9_clusters': 'K=9',
#     'truth_token_score': 'Truth Silhouette Score', 'pred_token_score': 'Prediction Silhouette Score'
# }
#
# distribution_map = {
#     'first_tokens_freq': 'First Token Frequency', 'first_tokens_consistency': 'First Token Consistency',
#     'first_tokens_inconsistency': 'First Token Inconsistency',
#     'losses': 'Loss', 'tokenization_rate': 'Tokenization Rate',
#     'token_entropy': 'Token Entropy', 'word_entropy': 'Word Entropy',
#     'prediction_entropy': 'Prediction Entropy', 'truth': 'Ground Truth',
#     'pred': 'Prediction', 'agreement': 'Class Agreement', 'error_type': 'Error Type',
#     'confidences': 'Confidence', 'variability': 'Variability',
#     'O': 'O Confidence', 'B-PERS': 'B-PERS Confidence', 'I-PERS': 'I-PERS Confidence', 'B-ORG': 'B-ORG Confidence',
#     'I-ORG': 'I-ORG Confidence',
#     'B-LOC': 'B-LOC Confidence', 'I-LOC': 'I-LOC Confidence', 'B-MISC': 'B-MISC Confidence',
#     'I-MISC': 'I-MISC Confidence',
#     '3_clusters': 'K=3', '4_clusters': 'K=4', '9_clusters': 'K=9',
#     'truth_token_score': 'Truth Silhouette Score', 'pred_token_score': 'Prediction Silhouette Score'
# }


def register_dataset_callbacks(app, dataset_obj):
    @app.callback(

        [
            Output('statistical_columns', 'options'),
            Output('distribution_column', 'options'),
            Output('categorical_column', 'options'),
            Output('correlation_columns', 'options'),
            Output('custom_distribution_selection', 'options'),
            Output('error_rate_columns', 'options'),
            Output('initialize_dataset', 'children'),
        ],

        Input("initialize_characteristics_tab", "n_clicks"),

    )
    def initialize_tab(n_clicks):
        if n_clicks > 0 and dataset_obj.loaded:
            summary_columns = dataset_obj.analysis_df.columns[10:]
            distribution_columns = dataset_obj.analysis_df.columns[10:]
            categorical_columns = list(dataset_obj.analysis_df.columns[10:-24]) + list(
                dataset_obj.analysis_df.columns[-3:])
            correlation_columns = list(dataset_obj.analysis_df.columns[16:]) + ['Error Rate']
            custom_distributions = [
                {'label': 'Tag Distribution', 'value': 'tag_distribution'},
                {'label': 'Tag Ambiguity', 'value': 'tag_ambiguity'},
                {'label': 'Token Length Distribution', 'value': 'token_length'},
                {'label': 'Sentence Length Distribution', 'value': 'sentence_length'},
            ]
            error_rate_columns = [
                'Anchor Token',
                'Anchor Token Frequency', 'Anchor Token Consistency', 'Anchor Token Inconsistency', 'Tokenization Rate',
            ]

            initialization_div = html.Div('Tab Initialized', style={'color': 'green'})
            return summary_columns, distribution_columns, categorical_columns, \
                   correlation_columns, custom_distributions, error_rate_columns, \
                   initialization_div

        else:
            raise PreventUpdate

    @app.callback(
        [
            Output("describe_table", "columns"),
            Output("describe_table", "data"),
        ],
        [
            Input('generate_summary', 'n_clicks'),
            State('statistical_columns', 'value'),
            State('include_ignored_stats', 'value'),
        ]
    )
    def update_describe_table(n_clicks, statistical_columns, include):
        if n_clicks > 0:
            if 'checked' in include:
                selected_df = dataset_obj.analysis_df.copy()

            else:
                selected_df = dataset_obj.analysis_df[dataset_obj.analysis_df['Label Id'] != -100]

            # Create a new DataFrame with the describe() output for the selected column
            selected_df = selected_df[statistical_columns]

            selected_df = selected_df.describe().reset_index().rename(columns={'index': 'Statistics'})
            # Rename columns for the DataTable
            selected_cols = [{'name': col, 'id': col} for col in selected_df.columns]

            # Convert the DataFrame to a dictionary of records for the DataTable
            selected_data = selected_df.to_dict('records')

            return selected_cols, selected_data
        else:
            raise PreventUpdate

    @app.callback(

        Output("distributions", "figure"),

        [
            Input('plot_distribution', 'n_clicks'),
            State('distribution_column', 'value'),
            State('categorical_column', 'value'),
            State('calculate_kde', 'value'),
        ]
    )
    def plot_distribution(n_clicks, distribution_column, categorical_column, calculate_kde):
        if n_clicks > 0:

            selected_df = dataset_obj.analysis_df[dataset_obj.analysis_df['Label Id'] != -100]
            try:
                if categorical_column is None:
                    # Create histogram using plotly.express
                    distribution_fig = px.histogram(selected_df, x=distribution_column, nbins=30, marginal="rug",
                                                    title=f'Distribution of {distribution_column}',
                                                    template='ggplot2',
                                                    )
                    distribution_fig.update_traces(marker=dict(line=dict(width=1.5, color='#FFFFFF')))

                    distribution_fig.update_layout(yaxis_title="Frequency")
                    if 'checked' in calculate_kde:
                        # Calculate KDE
                        kde = gaussian_kde(selected_df[distribution_column])
                        x_range = np.linspace(min(selected_df[distribution_column]),
                                              max(selected_df[distribution_column]),
                                              100)
                        y_kde = kde(x_range)

                        # Overlay KDE on the histogram
                        distribution_fig.add_trace(
                            go.Scatter(x=x_range, y=y_kde * len(selected_df) * np.diff(x_range)[0], mode='lines',
                                       name='KDE'))
                        # Set y-axis to log scale
                        distribution_fig.update_layout(yaxis_type="log", yaxis_title="Log Frequency")

                    return distribution_fig
                else:
                    complex_distribution = px.violin(selected_df, y=distribution_column, x=categorical_column, box=True,
                                                     points="all", template='ggplot2')

                    return complex_distribution
            except:
                raise PreventUpdate
        else:
            raise PreventUpdate

    @app.callback(

        Output("correlations", "figure"),

        [
            Input('calculate_correlation', 'n_clicks'),
            State('correlation_columns', 'value'),
            State('correlation_method', 'value'),
        ]
    )
    def calculate_correlation(n_clicks, correlation_columns, correlation_method):
        if n_clicks > 0:
            selected_df = dataset_obj.analysis_df[dataset_obj.analysis_df['Label Id'] != -100]
            selected_df['Error Rate'] = selected_df['Ground Truth'] != selected_df['Prediction']
            try:
                if correlation_method == "difference":
                    pearson_matrix = selected_df[correlation_columns].corr(method="pearson")
                    spearman_matrix = selected_df[correlation_columns].corr(method="spearman")
                    correlation_matrix = pearson_matrix - spearman_matrix
                elif correlation_method is not None:
                    correlation_matrix = selected_df[correlation_columns].corr(method=correlation_method)
                else:
                    correlation_matrix = selected_df[correlation_columns].corr(method='pearson')
                correlation_fig = px.imshow(
                    correlation_matrix,
                    labels=dict(x="Predicted Label", y="True Label", color="Correlation"),
                    title='Correlation Matrix',
                    width=800,  # Custom width
                    height=600  # Custom height
                )
                return correlation_fig

            except:
                raise PreventUpdate
        else:
            raise PreventUpdate

    @app.callback(

        Output("custom_distribution_output", "children"),

        [
            Input('custom_distribution', 'n_clicks'),
            State('custom_distribution_selection', 'value'),
        ]
    )
    def custom_distributions(n_clicks, custom_distribution_selection):
        if n_clicks > 0:
            selected_df = dataset_obj.analysis_df[dataset_obj.analysis_df['Label Id'] != -100]
            try:
                if custom_distribution_selection == 'tag_distribution':
                    counts = selected_df['Ground Truth'].value_counts().sort_index()
                    types = selected_df.groupby('Ground Truth')['Anchor Token'].nunique()
                    ratios = types / counts

                    tag_distribution_df = pd.DataFrame({
                        'Raw Counts': counts,
                        'Types': types,
                        'Count Type Ratio': ratios
                    })

                    totals = selected_df['Anchor Token'].agg(['size', 'nunique']).tolist()
                    ne_totals = selected_df[selected_df['Ground Truth'] != 'O']['Anchor Token'].agg(
                        ['size', 'nunique']).tolist()

                    tag_distribution_df.loc['Total'] = totals + [totals[1] / totals[0]]
                    tag_distribution_df.loc['Total NEs'] = ne_totals + [ne_totals[1] / ne_totals[0]]
                    tag_distribution_df = tag_distribution_df.rename(columns={'Raw Counts': 'Number of Tokens',
                                                                              'Types': 'Number of Types',
                                                                              'Count Type Ratio': 'Token-Type Ratio', })

                    tag_distribution_df['Number of Tokens'] = tag_distribution_df['Number of Tokens'].astype(int)
                    tag_distribution_df['Number of Types'] = tag_distribution_df['Number of Types'].astype(int)
                    tag_distribution_df['Token-Type Ratio'] = tag_distribution_df['Token-Type Ratio'].apply(
                        lambda x: round(x, 3))
                    tag_distribution_df = tag_distribution_df.sort_values(by='Number of Tokens', ascending=False)
                    tag_distribution_df = tag_distribution_df.reset_index().rename(
                        columns={'Ground Truth': 'Category'})

                    # Calculate the proportions for each category
                    tag_distribution_df['NEs Proportion'] = tag_distribution_df['Number of Tokens'] / ne_totals[0]

                    # Format the proportions to a more readable percentage format
                    tag_distribution_df['NEs Proportion'] = tag_distribution_df['NEs Proportion'].apply(
                        lambda x: round(x * 100, 2))

                    tag_distribution_table = dash_table.DataTable(
                        id='linguistic_diversity_table',
                        columns=[{"name": i, "id": i} for i in tag_distribution_df.columns],
                        style_header={'text-align': 'center', 'background-color': '#555555',
                                      'color': 'white'},
                        data=tag_distribution_df.to_dict('records'),
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
                        page_size=11,
                    ),
                    return tag_distribution_table

                elif custom_distribution_selection == 'tag_ambiguity':
                    tag_ambiguity_analysis = selected_df.groupby(['Ground Truth', 'Anchor Token']).agg(
                        mean_consistency=('Anchor Token Consistency', 'mean'),
                        mean_inconsistency=('Anchor Token Inconsistency', 'mean'),
                        mean_token_entropy=('Token Entropy', 'mean')
                    ).reset_index()
                    tag_ambiguity_summary = tag_ambiguity_analysis.groupby('Ground Truth').agg(
                        overall_mean_consistency=('mean_consistency', 'mean'),
                        overall_mean_inconsistency=('mean_inconsistency', 'mean'),
                        overall_mean_token_entropy=('mean_token_entropy', 'mean')
                    ).reset_index().round(3)
                    tag_ambiguity_table = dash_table.DataTable(
                        id='linguistic_diversity_table',
                        columns=[{"name": i, "id": i} for i in tag_ambiguity_summary.columns],
                        style_header={'text-align': 'center', 'background-color': '#555555',
                                      'color': 'white'},
                        data=tag_ambiguity_summary.to_dict('records'),
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
                        page_size=11,
                    ),
                    return tag_ambiguity_table

                elif custom_distribution_selection == 'token_length':
                    # Calculate the word length for each word in data_df
                    selected_df['Anchor Token Length'] = selected_df['Anchor Token'].apply(lambda x: len(str(x)))

                    # Use Plotly Express to generate the histogram
                    word_length_fig = px.histogram(selected_df, x='Anchor Token Length', nbins=30, marginal="box",
                                                   title='Distribution of Token Lengths', template='ggplot2')

                    # Add a vertical line for the average word length
                    word_length_fig.update_layout(
                        xaxis_title='Word Length',
                        yaxis_title='Frequency',
                    )

                    return dcc.Graph(figure=word_length_fig)

                else:
                    sentence_length_df = selected_df.groupby('Sentence Id')['Anchor Token'].count().reset_index()
                    sentence_length_df.columns = ['Sentence Id', 'Sentence Length']

                    sentence_length_simple_fig = px.histogram(sentence_length_df, x='Sentence Length', nbins=30,
                                                              marginal="box",
                                                              title='Distribution of Sentence Lengths',
                                                              template='ggplot2')

                    # Update the x-axis and y-axis titles
                    sentence_length_simple_fig.update_layout(
                        xaxis_title='Sentence Length (Number of Tokens)',
                        yaxis_title='Frequency'
                    )
                    sentence_length_simple_fig.update_traces(marker=dict(line=dict(width=1.5, color='#FFFFFF')))

                    return dcc.Graph(figure=sentence_length_simple_fig)

                return None

            except:
                raise PreventUpdate
        else:
            raise PreventUpdate

    @app.callback(

        Output("error_rate", "children"),

        [
            Input('calculate_error_rate', 'n_clicks'),
            State('error_rate_columns', 'value'),
        ]
    )
    def calculate_error_rate(n_clicks, error_rate_column):
        if n_clicks > 0:
            selected_df = dataset_obj.analysis_df[dataset_obj.analysis_df['Label Id'] != -100]
            selected_df['Errors'] = selected_df['Ground Truth'] != selected_df['Prediction']
            errors = selected_df[selected_df['Errors'] == True]
            if error_rate_column is None:
                total_errors_count = selected_df['Errors'].sum()
                overall_error_rate = selected_df['Errors'].mean()

                class_counts_summary = selected_df.groupby('Ground Truth')['Anchor Token'].count().reset_index()
                class_counts_summary.columns = ['Ground Truth', 'Class Counts']
                # class_counts = class_counts_summary['Anchor Token'].values

                # Calculate the error rate for each class
                class_error_rate = selected_df.groupby('Ground Truth')['Errors'].mean().reset_index().sort_values(
                    by='Errors', ascending=False)
                class_error_rate['Errors'] = class_error_rate['Errors'].apply(lambda x: f'{round(x * 100, 3)}%')

                class_error_counts = selected_df[selected_df['Errors'] == True][
                    'Ground Truth'].value_counts().reset_index()
                class_error_counts.columns = ['Ground Truth', 'Error Count']

                class_error_summary = pd.merge(class_error_rate, class_error_counts, on='Ground Truth',
                                               how='left').fillna(0)
                class_error_summary = pd.merge(class_error_summary, class_counts_summary, on='Ground Truth',
                                               how='left').fillna(0)
                class_error_summary['Error Proportion (Relative to Total Errors)'] = class_error_summary[
                                                                                         'Error Count'] / total_errors_count
                class_error_summary['Error Proportion (Relative to Total Errors)'] = class_error_summary[
                    'Error Proportion (Relative to Total Errors)'].apply(lambda x: f'{round(x * 100, 3)}%')

                class_error_summary[
                    ['Ground Truth', 'Class Counts', 'Error Count', 'Error Proportion (Relative to Total Errors)']]

                class_error_summary = class_error_summary.rename(columns={
                    'Error Count': 'Total Errors (Count)',
                    'Errors': 'Error Proportion (Relative to Class Size)',
                    'Error Proportion (Relative to Total Errors)': 'Error Proportion (Relative to Total Errors)'
                })

                # Selecting the relevant columns
                error_summary_table = class_error_summary[['Ground Truth', 'Class Counts', 'Total Errors (Count)',
                                                           'Error Proportion (Relative to Class Size)',
                                                           'Error Proportion (Relative to Total Errors)']]

                overall_error_data = {
                    'Ground Truth': 'Overall',
                    'Class Counts': len(selected_df),
                    'Total Errors (Count)': total_errors_count,
                    'Error Proportion (Relative to Class Size)': round(overall_error_rate * 100, 3),
                    'Error Proportion (Relative to Total Errors)': 1
                    # Since this is the overall error, the proportion relative to total errors is 1
                }
                error_summary_table = pd.concat([error_summary_table, pd.DataFrame([overall_error_data])],
                                                ignore_index=True)

                error_rate_summary_table = dash_table.DataTable(
                    id='error_summary_table',
                    columns=[{"name": i, "id": i} for i in error_summary_table.columns],
                    style_header={'text-align': 'center', 'background-color': '#555555',
                                  'color': 'white'},
                    data=error_summary_table.to_dict('records'),
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

                return error_rate_summary_table
            elif error_rate_column == 'Anchor Token':
                token_errors = selected_df.groupby(error_rate_column)['Errors'].sum().reset_index()
                columns_error_rate = errors.groupby(error_rate_column)[[
                    'Anchor Token Frequency', 'Anchor Token Consistency', 'Anchor Token Inconsistency',
                    'Tokenization Rate',
                    'Token Entropy', 'Prediction Entropy', 'Variability',
                    'Confidence', 'Loss',
                    'Truth Silhouette Score', 'Prediction Silhouette Score',
                ]].mean().reset_index().round(3)

                # Get the top 10 tokens with the highest error rates
                top_error_tokens = token_errors.sort_values(by='Errors', ascending=False)

                # Merge with the main data frame to get other characteristics of these tokens
                top_error_tokens = top_error_tokens.merge(columns_error_rate,
                                                          on=error_rate_column,
                                                          how='left')
                top_error_tokens_tables = dash_table.DataTable(
                    id='error_summary_table',
                    columns=[{"name": i, "id": i} for i in top_error_tokens.columns],
                    style_header={'text-align': 'center', 'background-color': '#555555',
                                  'color': 'white'},
                    data=top_error_tokens.to_dict('records'),
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

                return top_error_tokens_tables
            else:

                token_counts = selected_df.groupby(error_rate_column)['Anchor Token'].count().reset_index()
                token_counts.columns = [error_rate_column, 'Token Count']
                error_counts = errors.groupby(error_rate_column)['Anchor Token'].count().reset_index()
                error_counts.columns = [error_rate_column, 'Error Count']
                error_rate = pd.merge(token_counts, error_counts)
                error_rate['Error Rate'] = error_rate['Error Count'] / error_rate['Token Count']

                error_rate_fig = px.line(error_rate,
                                         x=error_rate_column,
                                         y='Error Rate',
                                         title=f"Error Rate vs {error_rate_column}",
                                         hover_data=['Token Count', 'Error Count'],
                                         markers=True
                                         )
                return dcc.Graph(figure=error_rate_fig)





        else:
            raise PreventUpdate
