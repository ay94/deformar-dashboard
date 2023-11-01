from utils.appUtils import DatasetConfig, Datasets
from utils.analysisUtils import FileHandler, AttentionSimilarity
from utils.plotUtils import get_input_trigger, default_coordinates, default_color, \
    default_entity, defualt_centroid, get_value, color_map, create_confusion_table, \
    compute_confusion, create_token_confusion, create_error_bars, extract_column, \
    identify_mistakes, color_tokens, min_max, default_view
from dash.dependencies import Input, Output, State
from dash import Dash, dash_table, dcc, html, callback_context
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

import pandas as pd
import numpy as np
import torch
import dash
from sklearn.metrics.pairwise import cosine_similarity
from bertviz import head_view, model_view
from tqdm import tqdm
# from .dataset_callback import *
# from .decision_callback import *

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

hover_data = [
    'Token Selector', 'Class Agreement', 'Confidence', 'Variability', 'Ground Truth', 'Prediction'
]

train_hover_data = [
    'Token Selector', 'Class Agreement', 'Ground Truth', 'Prediction'
]
