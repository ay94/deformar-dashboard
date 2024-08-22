import plotly.express as px
import plotly.graph_objects as go
import logging

from enum import Enum
from utils.layout_managers import CustomDataTable
import pandas as pd
from dataclasses import dataclass
from dash import html
from scipy.stats import gaussian_kde
import numpy as np

@dataclass(frozen=True)
class TokenDistributionColumns:
    TRUE_LABELS: str = 'True Labels'
    CORE_TOKENS: str = 'Core Tokens'
    RAW_COUNTS: str = 'Raw Counts'
    TYPES: str = 'Types'
    COUNT_TYPE_RATIO: str = 'Count Type Ratio'
    NUMBER_OF_TOKENS: str = 'Number of Tokens'
    NUMBER_OF_TYPES: str = 'Number of Types'
    TOKEN_TYPE_RATIO: str = 'Token-Type Ratio'
    CATEGORY: str = 'Category'
    NEs_PROPORTION: str = 'NEs Proportion'

@dataclass(frozen=True)
class TagAmbiguityColumns:
    TRUE_LABELS: str = 'True Labels'
    CORE_TOKENS: str = 'Core Tokens'
    CONSISTENCY: str = 'Consistency Count'
    INCONSISTENCY: str = 'Inconsistency Count'
    TOKEN_ENTROPY: str = 'Local Token Entropy'
    MEAN_CONSISTENCY: str = 'mean_consistency'
    MEAN_INCONSISTENCY: str = 'mean_inconsistency'
    MEAN_TOKEN_ENTROPY: str = 'mean_token_entropy'
    OVERALL_MEAN_CONSISTENCY: str = 'overall_mean_consistency'
    OVERALL_MEAN_INCONSISTENCY: str = 'overall_mean_inconsistency'
    OVERALL_MEAN_TOKEN_ENTROPY: str = 'overall_mean_token_entropy'


class ResultsType(Enum):
    TRAINING = 'Training Results'
    CLUSTERING = 'Clustering Results'

class DistributionsType(Enum):
    TOKEN = 'Token Distribution'
    TAG = 'Tag Ambiguity' 
    TOKEN_LENGTH = 'Token Length Distribution' 
    SENTENCE_LENGTH = 'Sentence Length Distribution' 

class DatasetTabManager:
    def __init__(self, variants_data):
        self.variants_data = variants_data

    def get_tab_data(self, variant):
        """Retrieve specific variant data."""
        return self.variants_data.get(variant, None)

    def filter_ignored(self, data):
        """Filter data based on a provided condition."""
        return data[data['Labels'] != -100] 
    
    def generate_statistics(self, data, columns):
        """Generate statistical summaries for given columns."""
        if not columns:
            return None

        valid_columns = [col for col in columns if col in data.columns]
        if not valid_columns:
            return None

        statistics_df = data[valid_columns].describe().reset_index().rename(columns={'index': 'Statistics'})
        return statistics_df

    def generate_summary_statistics(self, variant, statistical_columns):
        """Generates summary statistics and handles the output for the summary table."""
        tab_data = self.get_tab_data(variant)
        if not tab_data:
            return None  # Indicate that no data was found

        filtered_data = self.filter_ignored(tab_data.analysis_data)

        data = self.generate_statistics(filtered_data, statistical_columns)
        if data is None or data.empty:
            return None  # Indicate that the data is empty

        return CustomDataTable(
            table_id="summary_data_table",
            data=data.to_dict('records'),
            columns=[{"name": col, "id": col} for col in data.columns]
        ).render()
    # def generate_distribution_plot(self, data, distribution_column, categorical_column=None, kde=False):
    #     """Generate a distribution plot based on the provided columns."""
    #     if distribution_column and categorical_column:
    #         # Use violin plot when both distribution_column and categorical_column are provided
    #         complex_distribution = px.violin(
    #             data, 
    #             y=distribution_column, 
    #             x=categorical_column, 
    #             points="all",  # Shows all points
    #             box=True,  # Adds a box plot inside the violin
    #             title=f'Violin Plot of {distribution_column} by {categorical_column}',
    #             template='ggplot2'
    #         )
    #         complex_distribution.update_layout(
    #             yaxis_title=distribution_column,
    #             xaxis_title=categorical_column
    #         )
    #         return complex_distribution

    #     elif distribution_column:
    #         # Create histogram when only distribution_column is provided
    #         distribution_fig = px.histogram(
    #             data, 
    #             x=distribution_column, 
    #             nbins=30, 
    #             marginal="rug",
    #             title=f'Distribution of {distribution_column}',
    #             template='ggplot2'
    #         )
    #         distribution_fig.update_traces(marker=dict(line=dict(width=1.5, color='#FFFFFF')))
    #         distribution_fig.update_layout(yaxis_title="Frequency")

    #         return distribution_fig

    #     return None  # Return None if no valid plot is generated
    # def generate_violin_plot(self, data, distribution_column, categorical_column):
    #     """Generate a violin plot with matching theme and style."""
    #     violin_fig = px.violin(
    #         data, 
    #         y=distribution_column, 
    #         x=categorical_column, 
    #         points="all",  # Shows all points
    #         box=True,  # Adds a box plot inside the violin
    #         title=f'Violin Plot of {distribution_column} by {categorical_column}',
    #         template='plotly_dark'  # Using the same dark theme
    #     )

    #     # Update layout to match your preferred color theme
    #     violin_fig.update_traces(
    #         line_color='#3DAFA8',  # Use the same color as your button for lines
    #         box_line_color='#3DAFA8',  # Matching the box plot lines with the button color
    #         meanline_color='#3DAFA8',  # Matching the mean line color
    #     )
        
    #     violin_fig.update_layout(
    #         yaxis_title=distribution_column,
    #         xaxis_title=categorical_column,
    #         autosize=True,
    #         margin=dict(l=10, r=10, t=30, b=30),  # Adjust margins as needed
    #         font=dict(color="#FFFFFF"),  # Set font color to white for contrast
    #         plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent plot background
    #         paper_bgcolor="rgba(0, 0, 0, 0)"  # Transparent paper background
    #     )

    #     return violin_fig

    # def generate_distribution_plot(self, data, distribution_column):
    #     """Generate a distribution plot based on the provided columns.""" 
    #     # Create histogram when only distribution_column is provided
    #     try:
    #         distribution_fig = px.histogram(
    #             data, 
    #             x=distribution_column, 
    #             nbins=30, 
    #             marginal="rug",
    #             title=f'Distribution of {distribution_column}',
    #             template='seaborn'  # Use plotly_dark for dark mode
    #         )
    #         distribution_fig.update_traces(marker=dict(line=dict(width=1.5, color='#FFFFFF'), color='#3DAFA8'))
    #         distribution_fig.update_layout(yaxis_title="Frequency")
    #         kde=True
    #         if kde:
    #             kde = gaussian_kde(data[distribution_column].dropna())
    #             x_range = np.linspace(data[distribution_column].min(), data[distribution_column].max(), 1000)
    #             y_kde = kde(x_range)

    #             distribution_fig.add_trace(
    #                 go.Scatter(x=x_range, y=y_kde * len(data) * np.diff(x_range)[0], mode='lines',
    #                         name='KDE', line=dict(color='#FF8C00'))
    #             )
    #         return distribution_fig
    #     except Exception as e:
    #         logging.error(f"Failed to generate histogram: {str(e)}")
    #         return None

    
    # def generate_distribution_or_violin(self, variant, distribution_column, categorical_column):
    #     tab_data = self.get_tab_data(variant)
    #     selected_df = self.filter_ignored(tab_data.analysis_data)
        
    #     if not distribution_column or not tab_data:
    #         return html.Span("Please select a column and click 'Plot Distribution' to view plot.")
        
    #     figure = None
    #     if distribution_column and categorical_column:
    #         figure = self.generate_violin_plot(
    #             selected_df,
    #             distribution_column,
    #             categorical_column
    #         )
    #     else:
    #         figure = self.generate_distribution_plot(
    #             selected_df,
    #             distribution_column,
    #             categorical_column=categorical_column,
    #         )

    #     if figure:
    #         figure.update_layout(autosize=True)
    #     return figure
    
    def generate_violin_plot(self, data, distribution_column, categorical_column):
        """Generate a violin plot with matching theme and style."""
        logging.debug(f"Generating violin plot for {distribution_column} by {categorical_column}")
        try:
            violin_fig = px.violin(
                data, 
                y=distribution_column, 
                x=categorical_column, 
                points="all",  # Shows all points
                box=True,  # Adds a box plot inside the violin
                title=f'Violin Plot of {distribution_column} by {categorical_column}',
                template='plotly_white'  # Using the same dark theme
            )

            # Update layout to match your preferred color theme
            violin_fig.update_traces(
                line_color='#3DAFA8',  # Use the same color as your button for lines
                box_line_color='#3DAFA8',  # Matching the box plot lines with the button color
                meanline_color='#3DAFA8',  # Matching the mean line color
                marker=dict(color='#FF7F7F')  # Change the color of the points to match your theme

            )
            
            violin_fig.update_layout(
                yaxis_title=distribution_column,
                xaxis_title=categorical_column,
                autosize=True,
                margin=dict(l=10, r=10, t=30, b=30),  # Adjust margins as needed
                font=dict(color="#FFFFFF"),  # Set font color to white for contrast
                # plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent plot background
                # paper_bgcolor="rgba(0, 0, 0, 0)"  # Transparent paper background
            )

            return violin_fig
        except Exception as e:
            logging.error(f"Failed to generate violin plot: {str(e)}")
            return None
    def generate_distribution_plot(self, data, distribution_column, kde=True):
        """Generate a distribution plot based on the provided columns.""" 
        try:
            distribution_fig = px.histogram(
                data, 
                x=distribution_column, 
                nbins=30, 
                marginal="rug",
                title=f'Distribution of {distribution_column}',
                template='plotly_white'  # Use plotly_dark for dark mode
            )
            distribution_fig.update_traces(marker=dict(line=dict(width=1.5, color='#FFFFFF'), color='#3DAFA8'))
            distribution_fig.update_layout(yaxis_title="Frequency")

            if kde:
                kde = gaussian_kde(data[distribution_column].dropna())
                x_range = np.linspace(data[distribution_column].min(), data[distribution_column].max(), 1000)
                y_kde = kde(x_range)

                distribution_fig.add_trace(
                    go.Scatter(x=x_range, y=y_kde * len(data) * np.diff(x_range)[0], mode='lines',
                            name='KDE', line=dict(color='#FF7F7F'))  # Slightly orange-yellow color for KDE
                )
            return distribution_fig
        except Exception as e:
            logging.error(f"Failed to generate histogram: {str(e)}")
            return None

    # def generate_distribution_or_violin(self, variant, distribution_column, categorical_column):
    #     tab_data = self.get_tab_data(variant)
    #     selected_df = self.filter_ignored(tab_data.analysis_data)
        
    #     if not distribution_column or not tab_data:
    #         return html.Span("Please select a column and click 'Plot Distribution' to view plot.")
        
    #     figure = None
    #     if distribution_column and categorical_column:
    #         figure = self.generate_violin_plot(
    #             selected_df,
    #             distribution_column,
    #             categorical_column
    #         )
    #     else:
    #         figure = self.generate_distribution_plot(
    #             selected_df,
    #             distribution_column,
    #         )
    #     if figure:
    #         figure.update_layout(autosize=True)

    #     return None
    def generate_distribution_or_violin(self, variant, distribution_column, categorical_column):
        tab_data = self.get_tab_data(variant)
        selected_df = self.filter_ignored(tab_data.analysis_data)
        
        if not distribution_column or not tab_data:
            return html.Span("Please select a column and click 'Plot Distribution' to view plot.")
        
        figure = None
        if distribution_column and categorical_column:
            figure = self.generate_violin_plot(
                selected_df,
                distribution_column,
                categorical_column
            )
        else:
            figure = self.generate_distribution_plot(
                selected_df,
                distribution_column,
            )
        
        if figure:
            figure.update_layout(autosize=True)
            return figure

        return None




    # def get_results_data(self, tab_data, results_type):
    #     if results_type == ResultsType.TRAINING:
    #         return tab_data.results
    #     elif results_type == ResultsType.CLUSTERING:
    #         return tab_data.kmeans_results
    #     return None

    def get_results_data(self, variant, results_type):
        """Fetch results data for a specific variant and results type."""
        tab_data = self.get_tab_data(variant)
        
        if not tab_data:
            logging.error("No data available for the selected variant.")
            return None
        
        try:
            results_type_enum = ResultsType(results_type)
        except ValueError:
            logging.error("Invalid results type selected.")
            return None

        if results_type_enum == ResultsType.TRAINING:
            data = tab_data.results
        elif results_type_enum == ResultsType.CLUSTERING:
            data = tab_data.kmeans_results
        else:
            logging.error("Unknown results type.")
            return None

        if data is None or data.empty:
            logging.warning("No results data available for the selected criteria.")
            return None

        return CustomDataTable(
            table_id="results_data_table",
            data=data.to_dict('records'),
            columns=[{"name": col, "id": col} for col in data.columns]
        ).render()

    
    def generate_empty_plot(self, message="No data available"):
        """Generate an empty plot with a specified message."""
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title=message,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{
                "text": message,
                "xref": "paper", "yref": "paper",
                "showarrow": False,
                "font": {"size": 20},
                "align": "center"
            }],
            height=600,  # Set a default height
            margin=dict(l=10, r=10, t=30, b=30)  # Adjust margins as needed
        )
        return empty_fig
    
    
    def create_token_distribution_table(self, selected_df):
        columns = TokenDistributionColumns()
        counts = selected_df[columns.TRUE_LABELS].value_counts().sort_index()
        types = selected_df.groupby(columns.TRUE_LABELS)[columns.CORE_TOKENS].nunique()
        ratios = types / counts

        # Create the DataFrame
        token_distribution_df = pd.DataFrame({
            columns.RAW_COUNTS: counts,
            columns.TYPES: types,
            columns.COUNT_TYPE_RATIO: ratios
        })

        # Ensure tag_distribution_df is a DataFrame
        assert isinstance(token_distribution_df, pd.DataFrame), "token_distribution_df is not a DataFrame"

        totals = selected_df[columns.CORE_TOKENS].agg(['size', 'nunique']).tolist()
        ne_totals = selected_df[selected_df[columns.TRUE_LABELS] != 'O'][columns.CORE_TOKENS].agg(['size', 'nunique']).tolist()

        token_distribution_df.loc['Total'] = totals + [totals[1] / totals[0]]
        token_distribution_df.loc['Total NEs'] = ne_totals + [ne_totals[1] / ne_totals[0]]

        token_distribution_df = token_distribution_df.rename(columns={
            columns.RAW_COUNTS: columns.NUMBER_OF_TOKENS,
            columns.TYPES: columns.NUMBER_OF_TYPES,
            columns.COUNT_TYPE_RATIO: columns.TOKEN_TYPE_RATIO,
        })

        # Break down the operations with print statements
        token_distribution_df[columns.NUMBER_OF_TOKENS] = token_distribution_df[columns.NUMBER_OF_TOKENS].astype(int)

        token_distribution_df[columns.NUMBER_OF_TYPES] = token_distribution_df[columns.NUMBER_OF_TYPES].astype(int)

        token_distribution_df[columns.TOKEN_TYPE_RATIO] = token_distribution_df[columns.TOKEN_TYPE_RATIO].apply(lambda x: round(x, 3))

        token_distribution_df = token_distribution_df.sort_values(by=columns.NUMBER_OF_TOKENS, ascending=False)
        token_distribution_df = token_distribution_df.reset_index().rename(columns={columns.TRUE_LABELS: columns.CATEGORY})

        # Calculate the proportions for each category
        token_distribution_df[columns.NEs_PROPORTION] = token_distribution_df[columns.NUMBER_OF_TOKENS] / ne_totals[0]
        token_distribution_df[columns.NEs_PROPORTION] = token_distribution_df[columns.NEs_PROPORTION].apply(lambda x: round(x * 100, 2))

        # Use the CustomDataTable class
        return CustomDataTable(
            table_id="token_distribution_table",
            data=token_distribution_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in token_distribution_df.columns]
        ).render()
    
    def create_tag_ambiguity_table(self, selected_df):
        columns = TagAmbiguityColumns()

        # Perform the initial group by and aggregation
        try:
            tag_ambiguity_analysis = selected_df.groupby([columns.TRUE_LABELS, columns.CORE_TOKENS]).agg(
                mean_consistency=(columns.CONSISTENCY, 'mean'),
                mean_inconsistency=(columns.INCONSISTENCY, 'mean'),
                mean_token_entropy=(columns.TOKEN_ENTROPY, 'mean')
            ).reset_index()
        except Exception as e:
            logging.error("Error during initial mean calculation step: %s", e)
            return None
                
        # Further aggregation to summarize the results
        try:
            tag_ambiguity_summary = tag_ambiguity_analysis.groupby(columns.TRUE_LABELS).agg(
                overall_mean_consistency=(columns.MEAN_CONSISTENCY, 'mean'),
                overall_mean_inconsistency=(columns.MEAN_INCONSISTENCY, 'mean'),
                overall_mean_token_entropy=(columns.MEAN_TOKEN_ENTROPY, 'mean')
            ).reset_index().round(3)
            
            tag_ambiguity_summary = tag_ambiguity_summary.rename(columns={
                'overall_mean_consistency': 'Overall Mean Consistency',
                'overall_mean_inconsistency': 'Overall Mean Inconsistency',
                'overall_mean_token_entropy': 'Overall Mean Token Entropy'
            })
        except Exception as e:
            logging.error("Error during overall mean aggregation step: %s", e)
            return None

        if tag_ambiguity_summary.empty:
            logging.warning("No data available for the selected criteria.")
            return None

        # Use the CustomDataTable class to render the DataTable
        return CustomDataTable(
            table_id='tag_ambiguity_table',
            data=tag_ambiguity_summary.to_dict('records'),
            columns=[{"name": i, "id": i} for i in tag_ambiguity_summary.columns],
        ).render()
    
    def custom_distributions(self, custom_distribution_type, variant):
        

        try:
            distribution_type_enum = DistributionsType(custom_distribution_type)
        except ValueError:
            logging.error("Invalid distribution type selected.")
            return None

        tab_data = self.get_tab_data(variant)
        if not tab_data:
            logging.error("No tab data available.")
            return None

        selected_df = self.filter_ignored(tab_data.analysis_data)
        if selected_df.empty:
            logging.error("No relevant data available after filtering.")
            return None

        if distribution_type_enum == DistributionsType.TOKEN:
            result = self.create_token_distribution_table(selected_df)
        elif distribution_type_enum == DistributionsType.TAG:
            result = self.create_tag_ambiguity_table(selected_df)
        else:
            logging.error("Invalid distribution type.")
            result = None

        return result


    
    # def create_tag_ambiguity_table(self, selected_df):
    #     columns = TagAmbiguityColumns()
        

        
    #     # Perform the initial group by and aggregation
    #     try:
    #         tag_ambiguity_analysis = selected_df.groupby([columns.TRUE_LABELS, columns.CORE_TOKENS]).agg(
    #             mean_consistency=(columns.CONSISTENCY, 'mean'),
    #             mean_inconsistency=(columns.INCONSISTENCY, 'mean'),
    #             mean_token_entropy=(columns.TOKEN_ENTROPY, 'mean')
    #         ).reset_index()
    #     except Exception as e:
    #         logging.error(f"Error during initial mean calculation step: {e}")
    #         return html.Div(
    #             "Failed to calculate initial means for tag ambiguity analysis.",
    #             style={'color': 'red', 'textAlign': 'center', 'marginTop': '20px'}
    #         )
            
            

    #     # Further aggregation to summarize the results
    #     try:
    #         tag_ambiguity_summary = tag_ambiguity_analysis.groupby(columns.TRUE_LABELS).agg(
    #             overall_mean_consistency=(columns.MEAN_CONSISTENCY, 'mean'),
    #             overall_mean_inconsistency=(columns.MEAN_INCONSISTENCY, 'mean'),
    #             overall_mean_token_entropy=(columns.MEAN_TOKEN_ENTROPY, 'mean')
    #         ).reset_index().round(3)
    #     except Exception as e:
    #         logging.error(f"Error during overall mean aggregation step: {e}")
    #         return html.Div(
    #             "Failed to calculate overall means for tag ambiguity summary.",
    #             style={'color': 'red', 'textAlign': 'center', 'marginTop': '20px'}
    #         )
    #     if tag_ambiguity_summary.empty:
    #         return html.Div(
    #             "No data available for the selected criteria.",
    #             style={'color': 'red', 'textAlign': 'center', 'marginTop': '20px'}
    #         )
    #     # Use the CustomDataTable class to render the DataTable
    #     return CustomDataTable(
    #         table_id='tag_ambiguity_table',
    #         data=tag_ambiguity_summary.to_dict('records'),
    #         columns=[{"name": i, "id": i} for i in tag_ambiguity_summary.columns],
    #     ).render()


