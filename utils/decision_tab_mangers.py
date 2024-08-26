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
from utils.decision_plotting_managers import (
                                DistributionAnalysis,
                                TokenVariabilityAnalysis,
                                TagAmbiguityAnalysis,
                                TokenLengthPlot,
                                SentenceLengthPlot,
                                ErrorRateAnalysis,
                                CorrelationMatrix,
                                DecisionScatter
                            )

from utils.enums import (
    ResultsType,
    CustomAnalysisType,
    CorrelationCoefficients,
    DecisionType
    )

from utils.tab_managers import BaseTabManager

class DecisionTabManager(BaseTabManager):
    def __init__(self, variants_data):
        super().__init__(variants_data)
    
    def get_training_data(self, variant):
        """Fetch training data for a specific variant."""
        tab_data = self.get_tab_data(variant)
        
        if not tab_data:
            logging.error("No data available for the selected variant.")
            return None        
        try:
            data = tab_data.training_data
        except ValueError:
            logging.error("Invalid results type selected.")
            return None


        if data is None or data.empty:
            logging.warning("No training data available.")
            return None

        return True

    def generate_matrix(self, variant, correlation_method):
        """
        Calculate and return correlation matrix and scatter plot for selected data.
        """
        tab_data = self.get_tab_data(variant)
        
        if not tab_data:
            logging.error("No data available for the selected variant.")
            return None
        
        selected_df = self.filter_ignored(tab_data.analysis_data)
        if selected_df.empty:
            logging.error("No relevant data available after filtering.")
            return None
        
        try:
            coefficient_type_enum = CorrelationCoefficients(correlation_method)
        except ValueError:
            logging.error("Invalid coefficient selected.")
            return None

        if coefficient_type_enum == CorrelationCoefficients.PEARSON:
            coefficient = 'pearson'
        elif coefficient_type_enum == CorrelationCoefficients.SPEARMAN:
            coefficient = 'spearman'
        else:
            logging.error("Unknown coefficient.")
            return None
        
        correlation_analysis = CorrelationMatrix()
        
        
        return correlation_analysis.generate_matrix(selected_df, coefficient)
    
    
    def generate_scatter_plot(self, variant, decision_type, color_column):
        """
        Calculate and return correlation matrix and scatter plot for selected data.
        """
        tab_data = self.get_tab_data(variant)
        
        if not tab_data:
            logging.error("No data available for the selected variant.")
            return None
        
        selected_df = self.filter_ignored(tab_data.analysis_data)
        if selected_df.empty:
            logging.error("No relevant data available after filtering.")
            return None
        try:
            decision_type_enum = DecisionType(decision_type)
        except ValueError:
            logging.error("Invalid decision type selected.")
            return None

        if decision_type_enum == DecisionType.FINETUNED:
            x_column='X'
            y_column='Y'
        elif decision_type_enum == DecisionType.PRETRAINED:
            x_column='Pre X'
            y_column='Pre Y'
        else:
            logging.error("Unknown Model.")
            return None
        
        
        if  not color_column:
            logging.error("Please select color column.")
            
        decision_analysis = DecisionScatter()
        
        return decision_analysis.generate_plot(selected_df, x_column=x_column, y_column=y_column, color_column=color_column)
    
    

    
    
    

    