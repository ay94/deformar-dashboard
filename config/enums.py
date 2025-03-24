import logging
from dataclasses import dataclass, field

from enum import Enum
from dataclasses import dataclass, field


class ResultsType(Enum):
    TRAINING = "Training Results"
    CLUSTERING = "Clustering Results"
    TOKEN = "Token-Level Report"
    ENTITY = "Entity-Level Report"
    STRICT_ENTITY = "Entity-Level Strict Report"


class CustomAnalysisType(Enum):
    TOKEN = "Token Variability"
    WORD = "Word Variability"
    TAG = "Tag Ambiguity"
    TOKEN_LENGTH = "Token Length Distribution"
    SENTENCE_LENGTH = "Sentence Length Distribution"
    TOKENIZATION_ERROR_RATE = "Tokenization Error Rate Analysis"


class CorrelationCoefficients(Enum):
    PEARSON = "Pearson"
    SPEARMAN = "Spearman"


class DecisionType(Enum):
    PRETRAINED = "Pre-trained Model"
    FINETUNED = "Fine Tuned Model"


# Define Enums for column names
class TokenVariabilityColumns(Enum):
    TRUE_LABELS = "True Labels"
    CORE_TOKENS = "Core Tokens"
    RAW_COUNTS = "Raw Counts"
    TYPES = "Types"
    COUNT_TYPE_RATIO = "Count Type Ratio"
    NUMBER_OF_TOKENS = "Number of Tokens"
    NUMBER_OF_TYPES = "Number of Types"
    TOKEN_TYPE_RATIO = "Token-Type Ratio"
    CATEGORY = "Category"
    NEs_PROPORTION = "NEs Proportion"

    @staticmethod
    def rename_mapping():
        """Provide a mapping for renaming columns."""
        return {
            TokenVariabilityColumns.RAW_COUNTS.value: TokenVariabilityColumns.NUMBER_OF_TOKENS.value,
            TokenVariabilityColumns.TYPES.value: TokenVariabilityColumns.NUMBER_OF_TYPES.value,
            TokenVariabilityColumns.COUNT_TYPE_RATIO.value: TokenVariabilityColumns.TOKEN_TYPE_RATIO.value,
        }


class TagAmbiguityColumns(Enum):
    TRUE_LABELS = "True Labels"
    CORE_TOKENS = "Core Tokens"
    CONSISTENCY = "Consistency Ratio"
    INCONSISTENCY = "Inconsistency Ratio"
    TOKEN_ENTROPY = "Local Token Entropy"
    WORD_ENTROPY = "Local Word Entropy"
    DATASET_TOKEN_ENTROPY = "Dataset Token Entropy"
    DATASET_WORD_ENTROPY = "Dataset Word Entropy"

    @staticmethod
    def aggregation_mapping():
        """Provide a mapping for aggregating column names."""
        return {
            "overall_mean_consistency": "mean_consistency",
            "overall_mean_inconsistency": "mean_inconsistency",
            "overall_mean_token_entropy": "mean_token_entropy",
            "overall_mean_dataset_token_entropy": "mean_dataset_token_entropy",
            "overall_mean_word_entropy": "mean_word_entropy",
            "overall_mean_dataset_word_entropy": "mean_dataset_word_entropy",
        }

    @staticmethod
    def rename_view_mapping():
        """Provide a mapping for renaming to view-friendly column names."""
        return {
            "overall_mean_consistency": "Overall Mean Consistency",
            "overall_mean_inconsistency": "Overall Mean Inconsistency",
            "overall_mean_token_entropy": "Overall Mean Token Entropy",
            "overall_mean_dataset_token_entropy": "Overall Mean Dataset Token Entropy",
            "overall_mean_word_entropy": "Overall Mean Word Entropy",
            "overall_mean_dataset_word_entropy": "Overall Mean Dataset Word Entropy",
        }


class CustomAnalysisConfig(Enum):
    TOKEN_LENGTH = {
        "column": "Core Tokens",  # Use 'Core Tokens' instead of 'Anchor Token'
        "x_column": "Core Token Length",
        "title": "Distribution of Core Token Lengths",
        "xaxis_title": "Core Token Length",
        "yaxis_title": "Frequency",
        "plot_type": "kde_histogram",  # Specify the plot type if needed
    }
    SENTENCE_LENGTH = {
        "column": "Core Tokens",
        "groupby_column": "Sentence Ids",
        "x_column": "Sentence Length",
        "title": "Distribution of Sentence Lengths",
        "xaxis_title": "Sentence Length (Number of Tokens)",
        "yaxis_title": "Frequency",
        "plot_type": "histogram",
    }

    @staticmethod
    def get_plot_config(plot_name):
        try:
            return CustomAnalysisConfig[plot_name.upper()].value
        except KeyError:
            logging.error(f"Plot configuration for {plot_name} not found.")
            return None


class ErrorRateColumns(Enum):
    TRUE_LABELS = "True Labels"
    PRED_LABELS = "Pred Labels"
    CORE_TOKENS = "Core Tokens"
    TOKENIZATION_RATE = "Tokenization Rate"
    ERRORS = "Errors"
    TOTAL_COUNT = "total_count"
    ERROR_COUNT = "error_count"
    ERROR_RATE = "error_rate"
    WEIGHTED_ERROR_RATE = "Weighted Error Rate"


# class CorrelationColumns(Enum):
#     LOSSES = "Losses"
#     TRUE_TOKEN_SCORE = "True Token Score"
#     PRED_TOKEN_SCORE = "Pred Token Score"
#     CONSISTENCY_COUNT = "Consistency Count"
#     CONSISTENCY_RATIO = "Consistency Ratio"
#     INCONSISTENCY_COUNT = "Inconsistency Count"
#     INCONSISTENCY_RATIO = "Inconsistency Ratio"
#     LOCAL_TOKEN_ENTROPY = "Local Token Entropy"
#     NORMALIZED_TOKEN_ENTROPY= "Normalized Token Entropy"
#     DATASET_TOKEN_ENTROPY = "Dataset Token Entropy"
#     LOCAL_WORD_ENTROPY = "Local Word Entropy"
#     NORMALIZED_WORD_ENTROPY = "Normalized Word Entropy"
#     DATASET_WORD_ENTROPY = "Dataset Word Entropy"
#     TOKENIZATION_RATE = "Tokenization Rate"
#     PREDICTION_ENTROPY = "Prediction Entropy"
#     NORMALIZED_PREDICTION_ENTROPY = "Normalized Prediction Entropy"
#     TOKEN_CONFIDENCE = "Token Confidence"
#     VARIABILITY = "Variability"
#     B_LOC_CONFIDENCE = "B-LOC Confidence"
#     B_PERS_CONFIDENCE = "B-PERS Confidence"
#     B_ORG_CONFIDENCE = "B-ORG Confidence"
#     B_MISC_CONFIDENCE = "B-MISC Confidence"
#     TRUE_LABELS = "True Labels"

#     @staticmethod
#     def list_columns():
#         """
#         Returns a list of all column names for use in correlation analysis.
#         """
#         return [
#             col.value
#             for col in CorrelationColumns
#             if col != CorrelationColumns.TRUE_LABELS
#         ]


@dataclass
class CorrelationColumns:
    numeric_variables: list = field(default_factory=lambda: [
        "Token Ambiguity", "Word Ambiguity",
        "Consistency Ratio", "Inconsistency Ratio", 
        "Tokenization Rate",
        "Token Confidence", "Loss Values", "Prediction Uncertainty", 
        "True Silhouette", "Pred Silhouette"
    ])
    core_metrics: list = field(default_factory=lambda: [
        "Losses", "True Silhouette Score", "Pred Silhouette Score", "Consistency Count",
        "Consistency Ratio", "Inconsistency Count", "Inconsistency Ratio",
        "Local Token Entropy", "Normalized Token Entropy", "Dataset Token Entropy",
        "Local Word Entropy", "Normalized Word Entropy", "Dataset Word Entropy",
        "Tokenization Rate", "Prediction Entropy", "Normalized Prediction Entropy",
        "Token Confidence", "Variability"
    ])
    confidence_metrics: list = field(default_factory=lambda: [
        "B-LOC Confidence", "B-PER Confidence", "B-ORG Confidence", "B-MISC Confidence",
        "I-LOC Confidence", "I-PER Confidence", "I-ORG Confidence", "I-MISC Confidence",
        "O Confidence"
    ])
    # true_labels: list = field(default_factory=lambda: ["True Labels"])
    
    def get_columns(self, include_confidence=False):
        cols = self.numeric_variables
        # cols = self.core_metrics
        # if include_confidence:
        #     cols += self.confidence_metrics
        return cols


@dataclass
class DisplayColumns:
    # meta_columns: list = field(default_factory=lambda: [
    #     "Sentence Ids", "Token Positions", "Words", "Tokens",
    #     "Word Pieces", "Core Tokens", "Token Selector Id", "Token Ids", "Global Id"
    # ])
    meta_columns: list = field(default_factory=lambda: [
        "Global Id", "Words", "Tokens",
        "Token Selector Id", 
    ])
    categorical_columns: list = field(default_factory=lambda: [
        True Labels
Pred Labels
Agreements
K=3
K=4
K=9
Boundary Clusters
Entity Clusters
Token Clusters
Error Type

 
    ])
    metric_columns: list = field(default_factory=lambda: [
        "Token Ambiguity", "Word Ambiguity", "Consistency Ratio", "Inconsistency Ratio",
        "Tokenization Rate", "Token Confidence", "Loss Values", "Prediction Uncertainty",
        "True Silhouette", "Pred Silhouette"
    ])

    def get_columns(self, include_meta=True):
        if include_meta:
            return self.meta_columns + self.metric_columns
        return self.metric_columns
    
    def get_summary_view(self, include_meta=False):
        summary = self.metric_columns[:5]
        return self.meta_columns + summary if include_meta else summary


# class ColorMap(Enum):
#     B_LOC = 'darkgreen'
#     B_PERS = 'deepskyblue'
#     B_ORG = 'darkcyan'
#     B_MISC = 'palevioletred'
#     I_LOC = 'yellowgreen'
#     I_PERS = 'lightblue'
#     I_ORG = 'cyan'
#     I_MISC = 'violet'
#     O = 'saddlebrown'
#     LOC = 'darkgreen'
#     PERS = 'deepskyblue'
#     ORG = 'darkcyan'
#     MISC = 'palevioletred'
#     NOUN = 'darkgreen'
#     VERB = 'deepskyblue'
#     PN = 'darkcyan'
#     PRT = 'yellowgreen'
#     ADJ = 'lightblue'
#     ADV = 'cyan'
#     PRON = 'saddlebrown'
#     DSIL = 'violet'
#     CCONJ = 'turquoise'
#     ADP = 'darksalmon'
#     PUNCT = 'tomato'
#     DET = 'midnightblue'
#     X = 'olive'
#     AUX = 'limegreen'
#     NUM = 'slateblue'
#     PART = 'wheat'
#     SYM = 'firebrick'
#     PROPN = 'gold'
#     INTJ = 'lightseagreen'
#     IGNORED = 'grey'
#     CLS = 'grey'
#     SEP = 'grey'
#     SELECTED = 'black'

#     @classmethod
#     def get_color_map(cls):
#         """Return the color map as a dictionary."""
#         return {item.name.replace("_", "-"): item.value for item in cls}


class HoverColumns(Enum):
    WORDS = "Words"
    CORE_TOKENS = "Core Tokens"
    TRUE_LABELS = "True Labels"
    TOKEN_SELECTOR_ID = "Token Selector Id"
    TOKEN_CONFIDENCE = "Token Confidence"
    PRED_LABELS = "Pred Labels"
    AGREEMENTS = "Agreements"
    ERROR_TYPE = "Error Type"

    @staticmethod
    def list_columns():
        """
        Returns a list of all column names for use in selections or other operations,
        excluding specific items if necessary.
        """
        # If you need to exclude specific items, you can do so with a condition
        return [col.value for col in HoverColumns]


class SelectionPlotColumns(Enum):
    TRUE_LABELS = "True Labels"
    PRED_LABELS = "Pred Labels"


class TrainColumns(Enum):
    X_COLUMN = "X"
    Y_COLUMN = "Y"
    LOSSES = "Losses"
    GLOBAL_ID = "Global Id"
    TRUE_LABELS = "True Labels"

    @staticmethod
    def list_columns():
        """
        Returns a list of all column names for use in selections or other operations,
        excluding specific items if necessary.
        """
        # If you need to exclude specific items, you can do so with a condition
        return [col.value for col in TrainColumns]
    

class TrainColumns(Enum):
    X_COLUMN = "X"
    Y_COLUMN = "Y"
    LOSSES = "Losses"
    GLOBAL_ID = "Global Id"
    TRUE_LABELS = "True Labels"

    @staticmethod
    def list_columns():
        """
        Returns a list of all column names for use in selections or other operations,
        excluding specific items if necessary.
        """
        # If you need to exclude specific items, you can do so with a condition
        return [col.value for col in TrainColumns]

