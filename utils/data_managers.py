import logging
from tqdm.autonotebook import tqdm
from experiment_utils.utils import FileHandler
import pandas as pd
from flask_caching import Cache
import numpy as np

from dataclasses import dataclass, field
from typing import Dict, Any
import plotly.graph_objects as go

@dataclass
class DashboardData:
    analysis_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    kmeans_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    results: pd.DataFrame = field(default_factory=pd.DataFrame)
    entity_confusion_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    centroids_avg_similarity_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    attention_weights_similarity: go.Figure = field(default_factory=go.Figure)
    attention_similarity_matrix: go.Figure = field(default_factory=go.Figure)

    def __post_init__(self):
        # Round float columns to four decimal places
        self.round_floats(self.analysis_data)
        self.round_floats(self.kmeans_results)
        self.round_floats(self.results)

        # Convert list to string in the 'Word Pieces' column of analysis_data if it exists
        if 'Word Pieces' in self.analysis_data.columns:
            self.analysis_data['Word Pieces'] = self.analysis_data['Word Pieces'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else x
            )
        self.analysis_data['Normalized Token Entropy'] = DashboardData.normalized_entropy(self.analysis_data, 'Local Token Entropy', 'Token Max Entropy', 'Normalized Token Entropy')  # filling 0/0 division as it generates Nan
        self.analysis_data['Normalized Word Entropy'] = DashboardData.normalized_entropy(self.analysis_data, 'Local Token Entropy', 'Token Max Entropy', 'Normalized Token Entropy')  # filling 0/0 division as it generates Nan
    @staticmethod
    def round_floats(df):
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = df[col].round(4)
    
    def is_loaded(self, attribute):
        """Checks if the given attribute is loaded based on its type."""
        attr_value = getattr(self, attribute)
        if isinstance(attr_value, pd.DataFrame):
            return not attr_value.empty
        elif isinstance(attr_value, go.Figure):
            return len(attr_value.data) > 0  # Check if the figure has data
        return False  # Default case if the attribute type is unrecognized

    @staticmethod
    def from_dict(dict_data: Dict[str, Any]):
        return DashboardData(**dict_data)
    @staticmethod
    def normalized_entropy(df, col1, col2, new_col_name):
        return np.where(
            (df[col1] == 0) & (df[col2] == 0),  # Condition for both columns being 0
            0,  # Value if condition is true
            np.where(
                (df[col1] == -1) & (df[col2] == -1),  # Condition for both columns being -1
                -1,  # Value if condition is true
                df[col1] / df[col2]  # Default calculation
            )
        )

class DataLoader:
    def __init__(self, config_manager, variant_name):
        self.data_config = config_manager.data_config
        self.data_dir = config_manager.data_dir / variant_name
        self.dashboard_data = {}

    def load(self, file_name, file_config):
        file_handler = FileHandler(self.data_dir / file_config['folder'])
        file_type = file_config["format"]
        file_path = file_handler.file_path / f"{file_name}.{file_type}"

        try:
            if file_path.exists():
                # Load Plotly figures specifically
                if file_name in ["attention_weights_similarity", "attention_similarity_matrix"]:
                    return file_handler.read_plotly(file_path)

                # Handle regular JSON data files
                elif file_type == "json":
                    data = file_handler.read_json(file_path)
                    if "column_mappings" in file_config and file_config["column_mappings"]:
                        data = self.apply_column_mappings(data, file_config["column_mappings"])
                    return data
            else:
                logging.warning("File does not exist: %s", file_path)
        except Exception as e:
            logging.error("Failed to load data from %s: %s", file_path, e)
            return None
        
    def apply_column_mappings(self, data: pd.DataFrame, column_mappings: dict) -> pd.DataFrame:
        """ Rename columns in the DataFrame based on provided mappings. """
        return data.rename(columns=column_mappings)


    def load_all(self):
        
        logging.info("Loading Dashboard Data from  %s", self.data_dir)
        for file_name, file_config in tqdm(self.data_config.items()):
            self.dashboard_data[file_name] = self.load(file_name, file_config)
            
            


class DataManager:
    def __init__(self, config_manager, server) -> None:
        self.config_manager = config_manager
        self.variants = config_manager.variants
        self.cache = Cache(server, config={
            'CACHE_TYPE': 'filesystem',
            'CACHE_DIR': 'cache-directory',
            'CACHE_DEFAULT_TIMEOUT': 3600  # Cache timeout of 1 hour
        })
        self.cache.init_app(server)
        self.variants_data = self.load_all_variants_from_cache()
    
    def load_all_variants_from_cache(self):
        data = {}
        for variant in self.variants:
            cached_data = self.cache.get(variant)
            if cached_data:
                data[variant] = cached_data
        return data

    
    def load_variant(self, variant):
        """Loads data for a specific variant, with caching."""
        cached_data = self.cache.get(variant)
        if cached_data is None:
            loader = DataLoader(self.config_manager, variant)
            loader.load_all()
            data = DashboardData.from_dict(loader.dashboard_data)
            self.variants_data[variant] = data
            self.cache.set(variant, data)  # Cache the newly loaded data
            return data  # Return the new data
        self.variants_data[variant] = cached_data
        return cached_data  # Return the cached data if it was already loaded
    
    def load_data(self):
        """Loads data for all variants using the load_variant method for consistency."""
        for variant in self.variants:
            # Delegate the loading and caching to load_variant method
            self.variants_data[variant] = self.load_variant(variant)
        return self.variants_data

    # def is_data_loaded(self):
    #     """Checks if all variants have data loaded in the cache."""
    #     for variant in self.variants:
    #         if self.cache.get(variant) is None:
    #             return False  # Return False if any variant is not loaded
    #     return True  # Return True if all variants are loaded
    def is_any_variant_loaded(self):
        """
        Check if any variant is loaded in the cache.

        Returns:
            bool: True if at least one variant is loaded, False otherwise.
        """
        for variant in self.variants:
            if self.cache.get(variant) is not None:
                return True  # Return True if any variant is loaded
        return False  # Return False if no variants are loaded

