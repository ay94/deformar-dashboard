from . import Input, Output, State, html
from . import dash_table
from . import PreventUpdate
from . import callback_context, get_input_trigger
from . import px, go, make_subplots, np, pd, gaussian_kde, dcc
from . import Datasets
from . import FileHandler
from . import DatasetConfig
import logging
from utils.tab_managers import DatasetTabManager, ResultsType, DistributionsType

def register_callbacks(app, variants_data):
    data_manager = DatasetTabManager(variants_data)

    @app.callback(
        Output('loaded_tab_data', 'children'),
        Input('variant_selector', 'value')
    )
    def update_output(variant):
        tab_data = data_manager.get_tab_data(variant)
        if tab_data:
            return html.Span(f"Tab data loaded and Variant {variant} is chosen.", style={"color": "green"})
        else:
            return html.Span("Tab data is not available.", style={"color": "red"})
    
   
    @app.callback(
        [
            Output("summary_statistics_table", "columns"),
            Output("summary_statistics_table", "data"),
            Output("statistics_table_container", "style"),
            Output("no_summary_table", "style"),
        ],
        [
            Input('generate_summary', 'n_clicks'),
            State('statistical_columns', 'value'), 
            State('variant_selector', 'value')
        ]
    )
    def generate_summary_statistics(n_clicks, statistical_columns, variant):
        if n_clicks == 0:
            raise PreventUpdate

        tab_data = data_manager.get_tab_data(variant)
        if not tab_data:
            return [], [], {'display': 'none'}, {'display': 'block'}

        filtered_data = data_manager.filter_ignored(tab_data.analysis_data)
        columns, data = data_manager.generate_statistics(filtered_data, statistical_columns)
        
        if not data:
            return [], [], {'display': 'none'}, {'display': 'block'}
        
        return columns, data, {'display': 'block'}, {'display': 'none'}

    
    @app.callback(
        
        Output("distribution_container", "children"),
        [
            Input('plot_distribution', 'n_clicks'),
            State('distribution_column', 'value'),
            State('categorical_column', 'value'),
            State('variant_selector', 'value')
        ]
    )
    def plot_distribution(n_clicks, distribution_column, categorical_column, variant):
        if n_clicks == 0:
            raise PreventUpdate

        tab_data = data_manager.get_tab_data(variant)
        selected_df = data_manager.filter_ignored(tab_data.analysis_data)
        # Generate the plot
        figure = data_manager.generate_distribution_plot(
            selected_df,
            distribution_column,
            categorical_column=categorical_column,
        )

        if figure:
            figure.update_layout(
                autosize=True,
                # height=600,  # Explicit height to make sure the plot isn't too small
                # margin=dict(l=10, r=10, t=30, b=30)  # Adjust margins as needed
            )
            return [dcc.Graph(figure=figure)]
         
        
        return  [html.Span("Please select a column and click 'Plot Distribution' to view plot.")]

    
    @app.callback(
        [
            Output("results_data_table", "columns"),
            Output("results_data_table", "data"),
            Output("results_table_container", "style"),
            Output("no_results_table", "style"),
        ],
        [
            Input('view_results_type', 'n_clicks'),
            State('results_type', 'value'), 
            State('variant_selector', 'value')
        ]
    )
    def view_results(n_clicks, results_type, variant):
        if n_clicks == 0:
            raise PreventUpdate

        tab_data = data_manager.get_tab_data(variant)
        if not tab_data:
            return [], [], {'display': 'none'}, {'display': 'block'}
        
        # Convert the selected results_type to the Enum
        try:
            results_type_enum = ResultsType(results_type)
        except ValueError:
            # Handle the case where the results_type is not valid
            return [], [], {'display': 'none'}, {'display': 'block'}
        
        # Fetch the correct results data using the new method in the tab manager
        results_data = data_manager.get_results_data(tab_data, results_type_enum)
        
        if results_data is None or results_data.empty:
            return [], [], {'display': 'none'}, {'display': 'block'}
        
        columns = [{'name': col, 'id': col} for col in results_data.columns]
        data = results_data.to_dict('records')
        
        return columns, data, {'display': 'block'}, {'display': 'none'}
    
    @app.callback(
        Output("custom_distribution_graph_container", "children"),
        [
            Input('plot_custom_distribution', 'n_clicks'),
            State('custom_distribution_type', 'value'),
            State('variant_selector', 'value')
        ]
    )
    def custom_distributions(n_clicks, custom_distribution_type, variant):
        if n_clicks == 0:
            raise PreventUpdate

        
        # Convert the selected results_type to the Enum
        try:
            distribution_type_enum = DistributionsType(custom_distribution_type)
        except ValueError:
            # Handle the case where the results_type is not valid
            return html.Div(
                "Invalid distribution type selected. Please select a valid option.",
                style={'color': 'red', 'textAlign': 'center', 'marginTop': '20px'}
            )

        tab_data = data_manager.get_tab_data(variant)
        selected_df = data_manager.filter_ignored(tab_data.analysis_data)
        
        if not custom_distribution_type or not tab_data:
            return html.Div(
                "Please select a column and click 'Custom Distribution' to view data.",
                style={'color': 'red', 'textAlign': 'center', 'marginTop': '20px'}
            )
        
        try:
            if distribution_type_enum == DistributionsType.TOKEN:
                return data_manager.create_token_distribution_table(selected_df)

            elif distribution_type_enum == DistributionsType.TAG:
                return data_manager.create_tag_ambiguity_table(selected_df)

            elif distribution_type_enum == DistributionsType.TOKEN_LENGTH:
                selected_df['Anchor Token Length'] = selected_df['Anchor Token'].apply(lambda x: len(str(x)))

                word_length_fig = px.histogram(selected_df, x='Anchor Token Length', nbins=30, marginal="box",
                                            title='Distribution of Token Lengths', template='ggplot2')

                word_length_fig.update_layout(
                    xaxis_title='Word Length',
                    yaxis_title='Frequency',
                )

                return dcc.Graph(figure=word_length_fig)

            else:  # Assume 'sentence_length' if none of the above
                sentence_length_df = selected_df.groupby('Sentence Ids')['Anchor Token'].count().reset_index()
                sentence_length_df.columns = ['Sentence Ids', 'Sentence Length']

                sentence_length_simple_fig = px.histogram(sentence_length_df, x='Sentence Length', nbins=30,
                                                        marginal="box",
                                                        title='Distribution of Sentence Lengths',
                                                        template='ggplot2')

                sentence_length_simple_fig.update_layout(
                    xaxis_title='Sentence Length (Number of Tokens)',
                    yaxis_title='Frequency'
                )
                sentence_length_simple_fig.update_traces(marker=dict(line=dict(width=1.5, color='#FFFFFF')))

                return dcc.Graph(figure=sentence_length_simple_fig)

        except Exception as e:
            logging.error("Failed to generate custom distribution:%s", str(e))
            raise PreventUpdate

