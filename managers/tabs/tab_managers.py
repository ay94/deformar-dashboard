import logging

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

from config.config_managers import (BarConfig, MatrixConfig, ScatterConfig,
                                    ViolinConfig)
from layouts.managers.layout_managers import CustomDataTable


class BaseTabManager:
    def __init__(self, data):
        self.data = data

    def get_tab_data(self, key):
        """Retrieve specific data based on a key."""
        return self.data.get(key, None)

    def filter_ignored(self, data, label_col="Labels", ignore_label=-100):
        """Filter data based on a provided condition."""
        return data[data[label_col] != ignore_label]


def create_violin_plot(
    data, distribution_column, categorical_column, config: ViolinConfig
):
    fig = px.violin(
        data,
        y=distribution_column,
        x=categorical_column,
        points="all",
        box=True,
        title=config.title,
        template=config.template,
    )

    # Customize plot appearance
    fig.update_traces(
        line_color=config.line_color,
        box_line_color=config.box_line_color,
        meanline_color=config.meanline_color,
        marker=dict(color=config.marker_color),
    )

    fig.update_layout(
        yaxis_title=config.yaxis_title or distribution_column,
        xaxis_title=config.xaxis_title or categorical_column,
        autosize=config.autosize,
        margin=config.margin,
        font=dict(color=config.font_color),
        plot_bgcolor=config.plot_bgcolor,
        paper_bgcolor=config.paper_bgcolor,
    )

    return fig


def create_histogram_plot(data, distribution_column, config: BarConfig):
    fig = px.histogram(
        data,
        x=distribution_column,
        nbins=config.nbins,
        marginal="rug",
        title=config.title,
        template=config.template,
    )

    fig.update_traces(
        marker=dict(
            line=dict(width=1.5, color=config.kde_line_color), color=config.line_color
        )
    )

    fig.update_layout(
        yaxis_title=config.yaxis_title,
        xaxis_title=config.xaxis_title or distribution_column,
        autosize=config.autosize,
        margin=config.margin,
        font=dict(color=config.font_color),
        plot_bgcolor=config.plot_bgcolor,
        paper_bgcolor=config.paper_bgcolor,
    )

    if config.kde:
        kde_model = gaussian_kde(data[distribution_column].dropna())
        x_range = np.linspace(
            data[distribution_column].min(), data[distribution_column].max(), 1000
        )
        y_kde = kde_model(x_range)
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_kde * len(data) * np.diff(x_range)[0],
                mode="lines",
                name="KDE",
                line=dict(color=config.kde_line_color),
            )
        )

    return fig


def create_scatter_plot(data, x_column, y_column, config: ScatterConfig):
    fig = px.scatter(
        data,
        x=x_column,
        y=y_column,
        title=config.title,
        template=config.template,
        hover_data=config.hover_data,
    )

    # Update plot appearance to match the style of other plots
    fig.update_traces(
        line=dict(color=config.line_color, width=config.line_width),
        marker=dict(
            color=config.marker_color,
            size=config.marker_size,
            line=dict(width=1.5, color=config.marker_color),
        ),
    )

    fig.update_layout(
        xaxis_title=config.xaxis_title or x_column,
        yaxis_title=config.yaxis_title or y_column,
        autosize=config.autosize,
        margin=config.margin,
        font=dict(color=config.font_color),
    )

    return fig


def create_scatter_plot_with_color(
    data, x_column, y_column, color_column, symbol_column, config: ScatterConfig
):
    fig = px.scatter(
        data,
        x=x_column,
        y=y_column,
        color=color_column,
        symbol=symbol_column,
        title=config.title,
        template=config.template,
        hover_data=config.hover_data,
        custom_data=["Global Id"],
        color_discrete_map=config.color_discrete_map,
    )

    fig.update_traces(
        marker=dict(
            size=config.marker_size,
            opacity=config.marker_opacity,
            line=dict(
                width=config.line_width,
                color=config.line_color,
            ),
        ),
        selected=dict(
            marker=dict(
                size=config.selected_marker_size, opacity=config.selected_opacity
            )
        ),  # Selected state
        unselected=dict(
            marker=dict(
                size=config.unselected_marker_size, opacity=config.unselected_opacity
            )
        ),  # Unselected state
    )
    fig.update_layout(
        xaxis_title=config.xaxis_title or x_column,
        yaxis_title=config.yaxis_title or y_column,
        autosize=config.autosize,
        height=config.height,
        xaxis_visible=config.xaxis_visible,  # Hide the x axis
        yaxis_visible=config.yaxis_visible,  # Hide the y axis
        xaxis_showgrid=config.xaxis_showgrid,  # Hide x-axis grid lines
        yaxis_showgrid=config.yaxis_showgrid,  # Hide y-axis grid lines
    )
    return fig


def create_correlation_matrix_plot(correlation_matrix, config: MatrixConfig):
    fig = px.imshow(
        correlation_matrix,
        labels=dict(x=config.x, y=config.y, color=config.color),
        title=config.title,
        color_continuous_scale=config.color_continuous_scale,
    )

    # Update plot layout to match configuration
    fig.update_layout(
        template=config.template,
        autosize=config.autosize,
        margin=config.margin,
        font=dict(color=config.font_color),
        width=config.width,
        height=config.height,
        xaxis=config.xaxis,
        yaxis=config.yaxis,
    )
    return fig


def create_similarity_matrix_plot(similarity_matrix, config: MatrixConfig):
    fig = px.imshow(
        similarity_matrix,
        labels=dict(x=config.x, y="Entity Tags", color=config.color),
        title=config.title,
        color_continuous_scale=config.color_continuous_scale,
        x=similarity_matrix.columns,
        y=similarity_matrix.index,
        aspect="auto",  # auto adjusts to aspect ratio
    )

    # Update plot layout to match configuration
    fig.update_layout(
        template=config.template,
        autosize=config.autosize,
        margin=config.margin,
        font=dict(color=config.font_color),
        width=config.width,
        height=config.height,
        xaxis=config.xaxis,
        yaxis=config.yaxis,
    )
    return fig


def create_scatter_width_plot(data, x_column, y_column, config: ScatterConfig):
    fig = px.scatter(
        data,
        x=x_column,
        y=y_column,
        color=config.marker_color,  # or other conditional color parameters
        title=config.title,
        template=config.template,
        hover_data=config.hover_data,
        trendline="ols"
    )
    

    fig.update_traces(
        marker=dict(
            size=config.marker_size,
            line=dict(color=config.line_color, width=config.line_width),
        )
    )

    fig.update_layout(
        yaxis_title=config.yaxis_title or y_column,
        autosize=config.autosize,
        width=config.width,
        height=config.height,
        margin=config.margin,
        font=dict(color=config.font_color),
    )

    return fig


def create_bar_chart(
    data,
    title,
    xaxis_title,
    yaxis_title,
    width,
    height,
    color_column=None,
    color_map=None,
):
    """
    Create a bar chart using Plotly with the specified color and formatting.
    """
    # Create the bar chart with optional color mapping
    if color_column and color_map:
        fig = px.bar(data, color_discrete_map=color_map, orientation="h", title=title)
    else:
        fig = px.bar(data, orientation="h", title=title, template="plotly_white")

    # Update the layout to match the desired formatting
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white",
        width=width,  # Slightly larger width for better visibility
        height=height,
    )
    # Update the traces to set bar colors and borders
    fig.update_traces(marker=dict(line=dict(width=1.5, color="#FF7F7F")))

    return fig


class BasePlotting:
    def __init__(self, default_color="#3DAFA8"):
        self.default_color = default_color

    def create_histogram(self, data, x_column, title, xaxis_title, yaxis_title):
        """
        Create a histogram using Plotly with the specified default color for bars.
        """
        fig = px.histogram(
            data,
            x=x_column,
            nbins=30,
            marginal="box",
            title=title,
            template="plotly_white",
        )
        fig.update_traces(
            marker=dict(line=dict(width=1.5, color="#FF7F7F"), color=self.default_color)
        )
        fig.update_layout(
            xaxis_title=xaxis_title, yaxis_title=yaxis_title, template="plotly_white"
        )
        return fig

    def create_kde_histogram(
        self, data, x_column, title, xaxis_title, yaxis_title, kde_color=None
    ):
        """
        Create a histogram with a KDE overlay using Plotly. Uses default color if KDE color not specified.
        """
        fig = self.create_histogram(data, x_column, title, xaxis_title, yaxis_title)

        # Calculate KDE
        kde = gaussian_kde(data[x_column].dropna())
        x_range = np.linspace(data[x_column].min(), data[x_column].max(), 1000)
        y_kde = kde(x_range)

        # Add KDE line with specified or default color
        kde_plot_color = kde_color if kde_color else self.default_color
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_kde * len(data) * np.diff(x_range)[0],
                mode="lines",
                name="KDE",
                line=dict(color=kde_plot_color),
            )
        )
        return fig


class BaseAnalysis:
    def handle_errors(func):
        """Decorator for handling errors in analysis functions."""

        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in {func.__name__}: {e}")
                return None

        return wrapper

    def render_table(self, data, table_id):
        """Render a DataTable component."""
        if data is None or data.empty:
            return None

        return CustomDataTable(
            table_id=table_id,
            data=data.to_dict("records"),
            columns=[{"name": i, "id": i} for i in data.columns],
        ).render()
