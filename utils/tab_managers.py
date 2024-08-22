import plotly.express as px
import plotly.graph_objects as go
import logging

from enum import Enum
from utils.layout_managers import CustomDataTable
import pandas as pd
from dataclasses import dataclass
from dash import html

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

    def generate_statistics(self, data, columns):
        """Generate statistical summaries for given columns."""
        if not columns:
            return [], []

        valid_columns = [col for col in columns if col in data.columns]
        if not valid_columns:
            return [], []

        describe_df = data[valid_columns].describe().reset_index().rename(columns={'index': 'Statistics'})
        columns = [{'name': col, 'id': col} for col in describe_df.columns]
        data = describe_df.to_dict('records')

        return columns, data

    def filter_ignored(self, data):
        """Filter data based on a provided condition."""
        return data[data['Labels'] != -100] 
    
    def generate_distribution_plot(self, data, distribution_column, categorical_column=None, kde=False):
        """Generate a distribution plot based on the provided columns."""
        if distribution_column and categorical_column:
            # Use violin plot when both distribution_column and categorical_column are provided
            complex_distribution = px.violin(
                data, 
                y=distribution_column, 
                x=categorical_column, 
                points="all",  # Shows all points
                box=True,  # Adds a box plot inside the violin
                title=f'Violin Plot of {distribution_column} by {categorical_column}',
                template='ggplot2'
            )
            complex_distribution.update_layout(
                yaxis_title=distribution_column,
                xaxis_title=categorical_column
            )
            return complex_distribution

        elif distribution_column:
            # Create histogram when only distribution_column is provided
            distribution_fig = px.histogram(
                data, 
                x=distribution_column, 
                nbins=30, 
                marginal="rug",
                title=f'Distribution of {distribution_column}',
                template='ggplot2'
            )
            distribution_fig.update_traces(marker=dict(line=dict(width=1.5, color='#FFFFFF')))
            distribution_fig.update_layout(yaxis_title="Frequency")

            return distribution_fig

        return None  # Return None if no valid plot is generated

    def get_results_data(self, tab_data, results_type):
        if results_type == ResultsType.TRAINING:
            return tab_data.results
        elif results_type == ResultsType.CLUSTERING:
            return tab_data.kmeans_results
        return None

    
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
            logging.error(f"Error during initial mean calculation step: {e}")
            return html.Div(
                "Failed to calculate initial means for tag ambiguity analysis.",
                style={'color': 'red', 'textAlign': 'center', 'marginTop': '20px'}
            )
            
            

        # Further aggregation to summarize the results
        try:
            tag_ambiguity_summary = tag_ambiguity_analysis.groupby(columns.TRUE_LABELS).agg(
                overall_mean_consistency=(columns.MEAN_CONSISTENCY, 'mean'),
                overall_mean_inconsistency=(columns.MEAN_INCONSISTENCY, 'mean'),
                overall_mean_token_entropy=(columns.MEAN_TOKEN_ENTROPY, 'mean')
            ).reset_index().round(3)
        except Exception as e:
            logging.error(f"Error during overall mean aggregation step: {e}")
            return html.Div(
                "Failed to calculate overall means for tag ambiguity summary.",
                style={'color': 'red', 'textAlign': 'center', 'marginTop': '20px'}
            )
        if tag_ambiguity_summary.empty:
            return html.Div(
                "No data available for the selected criteria.",
                style={'color': 'red', 'textAlign': 'center', 'marginTop': '20px'}
            )
        # Use the CustomDataTable class to render the DataTable
        return CustomDataTable(
            table_id='tag_ambiguity_table',
            data=tag_ambiguity_summary.to_dict('records'),
            columns=[{"name": i, "id": i} for i in tag_ambiguity_summary.columns],
            style_header={'text-align': 'center', 'background-color': '#555555', 'color': 'white'},
            page_size=11
        ).render()


