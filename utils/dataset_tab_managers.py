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
from utils.plotting_managers import (
                                DistributionAnalysis,
                                TokenVariabilityAnalysis,
                                TagAmbiguityAnalysis,
                                TokenLengthPlot,
                                SentenceLengthPlot,
                                ErrorRateAnalysis,
                                CorrelationAnalysis
                            )

from utils.enums import (
    ResultsType,
    CustomAnalysisType,
    CorrelationCoefficients
    )

from utils.tab_managers import BaseTabManager

class DatasetTabManager(BaseTabManager):
    def __init__(self, variants_data):
        super().__init__(variants_data)
    
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
    


    def generate_distribution_or_violin(self, variant, distribution_column, categorical_column):
        tab_data = self.get_tab_data(variant)
        selected_df = self.filter_ignored(tab_data.analysis_data)
        
        if not distribution_column or not tab_data:
            return None
        
        # Delegate to PlottingAnalysis
        plotting_analysis = DistributionAnalysis()
        figure = plotting_analysis.generate_distribution_or_violin(selected_df, distribution_column, categorical_column)

        return figure


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
    
    def calculate_correlation(self, variant, correlation_method, x_column='Inconsistency Count', y_column = 'Inconsistency Count'):
        """
        Calculate and return correlation matrix and scatter plot for selected data.
        """
        tab_data = self.get_tab_data(variant)
        
        if not tab_data:
            logging.error("No data available for the selected variant.")
            return None, None
        
        selected_df = self.filter_ignored(tab_data.analysis_data)
        if selected_df.empty:
            logging.error("No relevant data available after filtering.")
            return None, None
        
        try:
            coefficient_type_enum = CorrelationCoefficients(correlation_method)
        except ValueError:
            logging.error("Invalid coefficient selected.")
            return None, None

        if coefficient_type_enum == CorrelationCoefficients.PEARSON:
            coefficient = 'pearson'
        elif coefficient_type_enum == CorrelationCoefficients.SPEARMAN:
            coefficient = 'spearman'
        else:
            logging.error("Unknown coefficient.")
            return None, None
        
        correlation_analysis = CorrelationAnalysis()
        
        
        return correlation_analysis.calculate_correlation(selected_df, coefficient, x_column, y_column)
    

        
    
    

    def create_token_variability_table(self, selected_df):
        """
        Calls the TokenDistributionAnalysis class to create a variability table.
        """
        token_variability_analysis = TokenVariabilityAnalysis()
        
        return token_variability_analysis.create_token_variability_table(selected_df)
    
    def create_tag_ambiguity_table(self, selected_df):
            """
            Calls the TagAmbiguityAnalysis class to create a tag ambiguity table.
            """
            tag_ambiguity_analysis = TagAmbiguityAnalysis()
            return tag_ambiguity_analysis.create_tag_ambiguity_table(selected_df)
    
    def create_token_length_plot(self, selected_df):
            """
            Calls the TagAmbiguityAnalysis class to create a tag ambiguity table.
            """
            plotter = TokenLengthPlot()
            figure = plotter.generate_plot(selected_df)
            return figure
    
    def create_sentence_length_plot(self, selected_df):
            """
            Calls the TagAmbiguityAnalysis class to create a tag ambiguity table.
            """
            plotter = SentenceLengthPlot()
            figure = plotter.generate_plot(selected_df)
            return figure
    
    def create_weighted_error_rate_plot(self, selected_df):
        """
        Calls the DistributionAnalysis class to create a weighted error rate plot.
        """
        error_rate_analysis = ErrorRateAnalysis()
        figure = error_rate_analysis.generate_weighted_error_rate_plot(selected_df,)
        return figure
    
    def perform_custom_analysis(self, custom_distribution_type, variant):
        

        try:
            analysis_type_enum = CustomAnalysisType(custom_distribution_type)
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

        if analysis_type_enum == CustomAnalysisType.TOKEN:
            
            analysis = self.create_token_variability_table(selected_df)
        elif analysis_type_enum == CustomAnalysisType.TAG:
            analysis = self.create_tag_ambiguity_table(selected_df)
        elif analysis_type_enum == CustomAnalysisType.TOKEN_LENGTH:
            analysis = self.create_token_length_plot(selected_df)
        elif analysis_type_enum == CustomAnalysisType.SENTENCE_LENGTH:
            analysis = self.create_sentence_length_plot(selected_df)
        elif analysis_type_enum == CustomAnalysisType.TOKENIZATION_ERROR_RATE:
            analysis = self.create_weighted_error_rate_plot(selected_df)
        else:
            logging.error("Invalid distribution type.")
            analysis = None

        return analysis


