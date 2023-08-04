import pandas as pd
from dash import Dash, html
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance
import json
import torch
import pickle as pkl
from torch import nn
import plotly.express as px
import numpy as np
from collections import defaultdict
import copy
import os
import time
import plotly.io as pio
from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor

from utils.analysisUtils import TokenAmbiguity, FineTuneConfig, TCDataset, TCModel

