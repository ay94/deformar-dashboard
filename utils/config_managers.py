from experiment_utils.utils import FileHandler
from experiment_utils.env_setup import init
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List
import yaml

@dataclass
class DevelopmentConfig:
    debug: bool = False
    port: int = 8000
    def __post_init__(self):
        if not isinstance(self.debug, bool):
            raise ValueError(f"Expected boolean for debug, got {type(self.debug).__name__}")
        if not (1 <= self.port <= 65535):
            raise ValueError("Port must be between 1 and 65535")
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]):
        return DevelopmentConfig(**config_dict)


@dataclass
class TabConfig:
    tab_value: str
    tab_label: str
    
@dataclass
class AppConfig:
    tabs: List[TabConfig] = field(default_factory=list)
    variants: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not all(isinstance(tab, TabConfig) for tab in self.tabs):
            raise ValueError("Tabs must be a list of TabConfig instances")
        if not all(isinstance(variant, str) for variant in self.variants):
            raise ValueError("Variants must be a list of strings")

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]):
        tabs = [TabConfig(**tab) for tab in config_dict.get('tabs', [])]
        variants = config_dict.get('variants', [])
        return AppConfig(tabs=tabs, variants=variants)

class DashboardConfigManager:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        config_fh = FileHandler(config_path.parent)
        try:
            self.config = config_fh.load_yaml(config_path.name)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file not found at {config_path}") from e
        except yaml.YAMLError as e:
            raise ValueError("Error parsing YAML configuration.") from e
        except ValueError as e:
            raise ValueError("Validation error in configuration.") from e

 
    @property
    def development_config(self) -> DevelopmentConfig:
        return DevelopmentConfig.from_dict(
            self.config.get("development", {})
        )

    @property
    def app_config(self) -> AppConfig:
        return AppConfig.from_dict(
            self.config.get("dashboard", {})
        )
    @property
    def data_dir(self) -> Path:
        base_folder = init()
        return base_folder / self.config.get("dashboard", {}).get('data_dir', '')

    @property
    def data_config(self) -> Dict:
        return self.config.get("dashboard", {}).get("dashboard_data", {}).get("data", {})

    @property
    def variants(self) -> Dict:
        return self.config.get("dashboard", {}).get("variants", {})
    
    @property
    def dataset_tab(self) -> Dict:
        return self.config.get("dashboard", {}).get("dataset_tab", {})



