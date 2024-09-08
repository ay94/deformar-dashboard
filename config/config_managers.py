from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml
from experiment_utils.env_setup import init
from experiment_utils.utils import FileHandler


@dataclass
class DevelopmentConfig:
    debug: bool = False
    port: int = 8000

    def __post_init__(self):
        if not isinstance(self.debug, bool):
            raise ValueError(
                f"Expected boolean for debug, got {type(self.debug).__name__}"
            )
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
        tabs = [TabConfig(**tab) for tab in config_dict.get("tabs", [])]
        variants = config_dict.get("variants", [])
        return AppConfig(tabs=tabs, variants=variants)


class DashboardConfigManager:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        config_fh = FileHandler(config_path.parent)
        try:
            self.config = config_fh.load_yaml(config_path.name)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Configuration file not found at {config_path}"
            ) from e
        except yaml.YAMLError as e:
            raise ValueError("Error parsing YAML configuration.") from e
        except ValueError as e:
            raise ValueError("Validation error in configuration.") from e

    @property
    def development_config(self) -> DevelopmentConfig:
        return DevelopmentConfig.from_dict(self.config.get("development", {}))

    @property
    def app_config(self) -> AppConfig:
        return AppConfig.from_dict(self.config.get("dashboard", {}))

    @property
    def data_dir(self) -> Path:
        base_folder = init()
        return base_folder / self.config.get("dashboard", {}).get("data_dir", "")

    @property
    def data_config(self) -> Dict:
        return (
            self.config.get("dashboard", {}).get("dashboard_data", {}).get("data", {})
        )

    @property
    def variants(self) -> Dict:
        return self.config.get("dashboard", {}).get("variants", {})

    @property
    def dataset_tab(self) -> Dict:
        return self.config.get("dashboard", {}).get("dataset_tab", {})

    @property
    def decision_tab(self) -> Dict:
        return self.config.get("dashboard", {}).get("decision_tab", {})


@dataclass
class ViolinConfig:
    title: str
    xaxis_title: str = ""
    yaxis_title: str = ""
    template: str = "plotly_white"
    line_color: str = "#000000"
    autosize: bool = True
    margin: dict = field(default_factory=lambda: dict(l=10, r=10, t=30, b=30))
    font_color: str = "#000000"
    plot_bgcolor: str = "rgba(0, 0, 0, 0)"
    paper_bgcolor: str = "rgba(0, 0, 0, 0)"
    box_line_color: str = "#000000"
    meanline_color: str = "#000000"
    marker_color: str = "#000000"


@dataclass
class BarConfig:
    title: str
    xaxis_title: str = ""
    yaxis_title: str = "Frequency"
    template: str = "plotly_white"
    line_color: str = "#000000"
    autosize: bool = True
    margin: dict = field(default_factory=lambda: dict(l=10, r=10, t=30, b=30))
    font_color: str = "#000000"
    plot_bgcolor: str = "rgba(0, 0, 0, 0)"
    paper_bgcolor: str = "rgba(0, 0, 0, 0)"
    nbins: int = 30
    kde_line_color: str = "#FF7F7F"
    kde: bool = False


@dataclass
class ScatterConfig:
    title: str
    xaxis_title: str = ""
    yaxis_title: str = ""
    template: str = "plotly_white"
    line_color: str = "#000000"
    marker_color: str = "#000000"
    marker_size: int = 10
    line_width: int = 2
    autosize: bool = True
    margin: dict = field(default_factory=lambda: dict(l=10, r=10, t=30, b=30))
    font_color: str = "#000000"
    hover_data: list = field(default_factory=list)
    color_discrete_map: dict = field(default_factory=dict)


@dataclass
class MatrixConfig:
    title: str
    x: str
    y: str
    color: str
    color_continuous_scale: str = "RdBu_r"
    template: str = "plotly_white"
    autosize: bool = True
    width: int = 700
    height: int = 700
    margin: dict = field(default_factory=lambda: dict(l=10, r=10, t=30, b=30))
    font_color: str = "#000000"
    xaxis: dict = field(default_factory=lambda: dict(showgrid=False, zeroline=False))
    yaxis: dict = field(default_factory=lambda: dict(showgrid=False, zeroline=False))


@dataclass
class ScatterWidthConfig:
    title: str
    xaxis_title: str = ""
    yaxis_title: str = ""
    template: str = "plotly_white"
    line_color: str = "#000000"
    marker_color: str = "#000000"
    marker_size: int = 10
    line_width: int = 2
    autosize: bool = True
    margin: dict = field(default_factory=lambda: dict(l=10, r=10, t=30, b=30))
    font_color: str = "#000000"
    hover_data: list = field(default_factory=list)
    color_discrete_map: dict = field(default_factory=dict)
    width: int = 700  # Specific to this plot
    height: int = 700  # Specific to this plot


# size=10,
# opacity=0.8
# size=5,
# opacity=0.9


@dataclass
class DecisionScatterConfig(ScatterConfig):
    # New fields
    marker_size: float = 5
    marker_opacity: int = 1.0
    line_width: float = 0.5
    line_color: str = "rgba(47, 79, 79, 1.0)"
    autosize: bool = True
    selected_marker_size: int = 10
    selected_opacity: float = 0.78
    unselected_marker_size: int = 5
    unselected_opacity: float = 0.9
    width: int = None
    height: int = None
    xaxis_visible: bool = False
    yaxis_visible: bool = False
    xaxis_showgrid: bool = False
    yaxis_showgrid: bool = False


@dataclass
class ColorMap:
    # Define a dictionary to hold color mappings
    color_map: Dict[str, str] = field(
        default_factory=lambda: {
            "B-LOC": "darkgreen",
            "B-PERS": "deepskyblue",
            "B-ORG": "darkcyan",
            "B-MISC": "palevioletred",
            "I-LOC": "yellowgreen",
            "I-PERS": "lightblue",
            "I-ORG": "cyan",
            "I-MISC": "violet",
            "O": "saddlebrown",
            "LOC": "darkgreen",
            "PERS": "deepskyblue",
            "ORG": "darkcyan",
            "MISC": "palevioletred",
            "NOUN": "darkgreen",
            "VERB": "deepskyblue",
            "PN": "darkcyan",
            "PRT": "yellowgreen",
            "ADJ": "lightblue",
            "ADV": "cyan",
            "PRON": "saddlebrown",
            "DSIL": "violet",
            "CCONJ": "turquoise",
            "ADP": "darksalmon",
            "PUNCT": "tomato",
            "DET": "midnightblue",
            "X": "olive",
            "AUX": "limegreen",
            "NUM": "slateblue",
            "PART": "wheat",
            "SYM": "firebrick",
            "PROPN": "gold",
            "INTJ": "lightseagreen",
            "IGNORED": "grey",
            "[CLS]": "grey",  # Explicitly handle CLS
            "[SEP]": "grey",  # Explicitly handle SEP
            "SELECTED": "black",
        }
    )

    def get_color(self, key: str) -> str:
        """Return the color for a given key, defaulting to 'grey' if not found."""
        return self.color_map.get(key, "grey")
