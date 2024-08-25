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

class DecisionTabManager(BaseTabManager):
    def __init__(self, variants_data):
        super().__init__(variants_data)
    

    