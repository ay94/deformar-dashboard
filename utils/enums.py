from enum import Enum
import logging


class ResultsType(Enum):
    TRAINING = 'Training Results'
    CLUSTERING = 'Clustering Results'

class CustomAnalysisType(Enum):
    TOKEN = 'Token Variability'
    TAG = 'Tag Ambiguity' 
    TOKEN_LENGTH = 'Token Length Distribution' 
    SENTENCE_LENGTH = 'Sentence Length Distribution'
    TOKENIZATION_ERROR_RATE = 'Tokenization Error Rate Analysis'
    

class CorrelationCoefficients(Enum):
    PEARSON = 'Pearson'
    SPEARMAN = 'Spearman' 
   
class DecisionType(Enum):
    PRETRAINED = 'Pre-trained Model'
    FINETUNED = 'Fine Tuned Model' 

# Define Enums for column names
class TokenVariabilityColumns(Enum):
    TRUE_LABELS = 'True Labels'
    CORE_TOKENS = 'Core Tokens'
    RAW_COUNTS = 'Raw Counts'
    TYPES = 'Types'
    COUNT_TYPE_RATIO = 'Count Type Ratio'
    NUMBER_OF_TOKENS = 'Number of Tokens'
    NUMBER_OF_TYPES = 'Number of Types'
    TOKEN_TYPE_RATIO = 'Token-Type Ratio'
    CATEGORY = 'Category'
    NEs_PROPORTION = 'NEs Proportion'
    
    @staticmethod
    def rename_mapping():
        """Provide a mapping for renaming columns."""
        return {
            TokenVariabilityColumns.RAW_COUNTS.value: TokenVariabilityColumns.NUMBER_OF_TOKENS.value,
            TokenVariabilityColumns.TYPES.value: TokenVariabilityColumns.NUMBER_OF_TYPES.value,
            TokenVariabilityColumns.COUNT_TYPE_RATIO.value: TokenVariabilityColumns.TOKEN_TYPE_RATIO.value,
        }


class TagAmbiguityColumns(Enum):
    TRUE_LABELS = 'True Labels'
    CORE_TOKENS = 'Core Tokens'
    CONSISTENCY = 'Consistency Count'
    INCONSISTENCY = 'Inconsistency Count'
    TOKEN_ENTROPY = 'Local Token Entropy'
    WORD_ENTROPY = 'Local Word Entropy'
    DATASET_TOKEN_ENTROPY = 'Dataset Token Entropy'
    DATASET_WORD_ENTROPY = 'Dataset Word Entropy'
    
    @staticmethod
    def aggregation_mapping():
        """Provide a mapping for aggregating column names."""
        return {
            'overall_mean_consistency': 'mean_consistency',
            'overall_mean_inconsistency': 'mean_inconsistency',
            'overall_mean_token_entropy': 'mean_token_entropy',
            'overall_mean_dataset_token_entropy': 'mean_dataset_token_entropy',
            'overall_mean_word_entropy': 'mean_word_entropy',
            'overall_mean_dataset_word_entropy': 'mean_dataset_word_entropy',
        }


    @staticmethod
    def rename_view_mapping():
        """Provide a mapping for renaming to view-friendly column names."""
        return {
            'overall_mean_consistency': 'Overall Mean Consistency',
            'overall_mean_inconsistency': 'Overall Mean Inconsistency',
            'overall_mean_token_entropy': 'Overall Mean Token Entropy',
            'overall_mean_dataset_token_entropy': 'Overall Mean Dataset Token Entropy',
            'overall_mean_word_entropy': 'Overall Mean Word Entropy',
            'overall_mean_dataset_word_entropy': 'Overall Mean Dataset Word Entropy'
        }



class CustomAnalysisConfig(Enum):
    TOKEN_LENGTH = {
        "column": "Core Tokens",  # Use 'Core Tokens' instead of 'Anchor Token'
        "x_column": "Core Token Length",
        "title": "Distribution of Core Token Lengths",
        "xaxis_title": "Core Token Length",
        "yaxis_title": "Frequency",
        "plot_type": "kde_histogram"  # Specify the plot type if needed
    }
    SENTENCE_LENGTH = {
        "column": "Core Tokens",
        "groupby_column": "Sentence Ids",
        "x_column": "Sentence Length",
        "title": "Distribution of Sentence Lengths",
        "xaxis_title": "Sentence Length (Number of Tokens)",
        "yaxis_title": "Frequency",
        "plot_type": "histogram"
    }

    @staticmethod
    def get_plot_config(plot_name):
        try:
            return CustomAnalysisConfig[plot_name.upper()].value
        except KeyError:
            logging.error(f"Plot configuration for {plot_name} not found.")
            return None



class ErrorRateColumns(Enum):
    TRUE_LABELS = 'True Labels'
    PRED_LABELS = 'Pred Labels'
    CORE_TOKENS = 'Core Tokens'
    TOKENIZATION_RATE = 'Tokenization Rate'
    ERRORS = 'Errors'
    TOTAL_COUNT = 'total_count'
    ERROR_COUNT = 'error_count'
    ERROR_RATE = 'error_rate'
    WEIGHTED_ERROR_RATE = 'weighted_error_rate'




class CorrelationColumns(Enum):
    LOSSES = 'Losses'
    TRUE_TOKEN_SCORE = 'True Token Score'
    PRED_TOKEN_SCORE = 'Pred Token Score'
    CONSISTENCY_RATIO = 'Consistency Count'
    INCONSISTENCY_RATIO = 'Inconsistency Count'
    LOCAL_TOKEN_ENTROPY = 'Local Token Entropy'
    TOKEN_MAX_ENTROPY = 'Token Max Entropy'
    DATASET_TOKEN_ENTROPY = 'Dataset Token Entropy'
    NORMALIZED_TOKEN_ENTROPY = 'Normalized Token Entropy'
    LOCAL_WORD_ENTROPY = 'Local Word Entropy'
    WORD_MAX_ENTROPY = 'Word Max Entropy'
    DATASET_WORD_ENTROPY = 'Dataset Word Entropy'
    NORMALIZED_WORD_ENTROPY = 'Normalized Word Entropy'
    TOKENIZATION_RATE = 'Tokenization Rate'
    PREDICTION_ENTROPY = 'Prediction Entropy'
    PREDICTION_MAX_ENTROPY = 'Prediction Max Entropy'
    TOKEN_CONFIDENCE = 'Token Confidence'
    VARIABILITY = 'Variability'
    TRUE_LABELS = 'True Labels'

    @staticmethod
    def list_columns():
        """
        Returns a list of all column names for use in correlation analysis.
        """
        return [col.value for col in CorrelationColumns if col != CorrelationColumns.TRUE_LABELS]


from enum import Enum

class ColorMap(Enum):
    B_LOC = 'darkgreen'
    B_PERS = 'deepskyblue'
    B_ORG = 'darkcyan'
    B_MISC = 'palevioletred'
    I_LOC = 'yellowgreen'
    I_PERS = 'lightblue'
    I_ORG = 'cyan'
    I_MISC = 'violet'
    O = 'saddlebrown'
    LOC = 'darkgreen'
    PERS = 'deepskyblue'
    ORG = 'darkcyan'
    MISC = 'palevioletred'
    NOUN = 'darkgreen'
    VERB = 'deepskyblue'
    PN = 'darkcyan'
    PRT = 'yellowgreen'
    ADJ = 'lightblue'
    ADV = 'cyan'
    PRON = 'saddlebrown'
    DSIL = 'violet'
    CCONJ = 'turquoise'
    ADP = 'darksalmon'
    PUNCT = 'tomato'
    DET = 'midnightblue'
    X = 'olive'
    AUX = 'limegreen'
    NUM = 'slateblue'
    PART = 'wheat'
    SYM = 'firebrick'
    PROPN = 'gold'
    INTJ = 'lightseagreen'
    IGNORED = 'grey'
    CLS = 'grey'
    SEP = 'grey'
    SELECTED = 'black'

    @classmethod
    def get_color_map(cls):
        """Return the color map as a dictionary."""
        return {item.name.replace("_", "-"): item.value for item in cls}

# Example of accessing the color map
custom_color_map = ColorMap.get_color_map()
