
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import gaussian_kde
from utils.config_managers import ViolinConfig, BarConfig
class BaseTabManager:
    def __init__(self, data):
        self.data = data

    def get_tab_data(self, key):
        """Retrieve specific data based on a key."""
        return self.data.get(key, None)

    def filter_ignored(self, data, label_col='Labels', ignore_label=-100):
        """Filter data based on a provided condition."""
        return data[data[label_col] != ignore_label].copy()




def create_violin_plot(data, distribution_column, categorical_column, config: ViolinConfig):
    fig = px.violin(
        data,
        y=distribution_column,
        x=categorical_column,
        points="all",
        box=True,
        title=config.title,
        template=config.template
    )

    # Customize plot appearance
    fig.update_traces(
        line_color=config.line_color,
        box_line_color=config.box_line_color,
        meanline_color=config.meanline_color,
        marker=dict(color=config.marker_color)
    )

    fig.update_layout(
        yaxis_title=config.yaxis_title or distribution_column,
        xaxis_title=config.xaxis_title or categorical_column,
        autosize=config.autosize,
        margin=config.margin,
        font=dict(color=config.font_color),
        plot_bgcolor=config.plot_bgcolor,
        paper_bgcolor=config.paper_bgcolor
    )

    return fig

def create_histogram_plot(data, distribution_column, config: BarConfig):
    fig = px.histogram(
        data,
        x=distribution_column,
        nbins=config.nbins,
        marginal="rug",
        title=config.title,
        template=config.template
    )

    fig.update_traces(marker=dict(line=dict(width=1.5, color=config.kde_line_color), color=config.line_color))

    fig.update_layout(
        yaxis_title=config.yaxis_title,
        xaxis_title=config.xaxis_title or distribution_column,
        autosize=config.autosize,
        margin=config.margin,
        font=dict(color=config.font_color),
        plot_bgcolor=config.plot_bgcolor,
        paper_bgcolor=config.paper_bgcolor
    )

    if config.kde:
        kde_model = gaussian_kde(data[distribution_column].dropna())
        x_range = np.linspace(data[distribution_column].min(), data[distribution_column].max(), 1000)
        y_kde = kde_model(x_range)
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_kde * len(data) * np.diff(x_range)[0],
                mode='lines',
                name='KDE',
                line=dict(color=config.kde_line_color)
            )
        )

    return fig


# def apply_common_styles(fig, config: PlotConfig):
#     fig.update_layout(
#         xaxis_title=config.xaxis_title,
#         yaxis_title=config.yaxis_title,
#         autosize=config.autosize,
#         margin=config.margin,
#         font=dict(color=config.font_color),
#         plot_bgcolor=config.bgcolor,
#         paper_bgcolor=config.bgcolor
#     )
#     fig.update_traces(line=dict(color=config.line_color))
