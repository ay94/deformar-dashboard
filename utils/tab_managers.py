
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import gaussian_kde
from utils.layout_managers import CustomDataTable
from utils.config_managers import ViolinConfig, BarConfig, ScatterConfig, MatrixConfig, ScatterWidthConfig
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



def create_scatter_plot(data, x_column, y_column, config: ScatterConfig):
    fig = px.scatter(
        data,
        x=x_column,
        y=y_column,
        title=config.title,
        template=config.template,
        hover_data=config.hover_data
    )

    # Update plot appearance to match the style of other plots
    fig.update_traces(
        line=dict(color=config.line_color, width=config.line_width),
        marker=dict(
            color=config.marker_color,
            size=config.marker_size,
            line=dict(width=1.5, color=config.marker_color)
        )
    )

    fig.update_layout(
        xaxis_title=config.xaxis_title or x_column,
        yaxis_title=config.yaxis_title or y_column,
        autosize=config.autosize,
        margin=config.margin,
        font=dict(color=config.font_color)
    )

    return fig



def create_scatter_plot_with_color(data, x_column, y_column, color_column, config: ScatterConfig):
    fig = px.scatter(
        data,
        x=x_column,
        y=y_column,
        color=color_column,
        title=config.title,
        template=config.template,
        hover_data=config.hover_data,
        color_discrete_map=config.color_discrete_map
    )

    # Update plot appearance to match the style of other plots
    # fig.update_traces(
    #     line=dict(color=config.line_color, width=config.line_width),
    #     marker=dict(
    #         size=config.marker_size,
    #         line=dict(width=1.5, color=config.line_color)
    #     )
    # )
#     fig.update_traces(
#     marker=dict(
#         size=config.marker_size,
#         line=dict(width=0)  # Set width to 0 to remove the border
#     )
# )

    fig.update_traces(
        marker=dict(
            size=5,
            line=dict(
                width=0.5,
                # color='DarkSlateGrey'
                color='rgba(47, 79, 79, 1.0)'

            )
        ),
        selected=dict(marker=dict(size=10, opacity=0.9)),  # Selected state
        unselected=dict(marker=dict(opacity=0.5))  # Unselected state
        # selector=dict(mode='markers')
    )
    fig.update_layout(
        xaxis_title=config.xaxis_title or x_column,
        yaxis_title=config.yaxis_title or y_column,
        autosize=config.autosize,
        height=500,
        xaxis_visible=False,  # Hide the x axis
        yaxis_visible=False,  # Hide the y axis
        xaxis_showgrid=False,  # Hide x-axis grid lines
        yaxis_showgrid=False,  # Hide y-axis grid lines
        # margin=config.margin,
        # margin=dict(l=3, r=3, t=20, b=20),
        # font=dict(color=config.font_color)
    )

    return fig

def create_matrix_plot(correlation_matrix, config: MatrixConfig):
    fig = px.imshow(
        correlation_matrix,
        labels=dict(x="Variables", y="Variables", color="Correlation"),
        title=config.title,
        color_continuous_scale=config.color_continuous_scale
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
        yaxis=config.yaxis
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
        hover_data=config.hover_data
    )

    fig.update_traces(
        marker=dict(
            size=config.marker_size,
            line=dict(color=config.line_color, width=config.line_width)
        )
    )

    fig.update_layout(
        xaxis_title=config.xaxis_title or x_column,
        yaxis_title=config.yaxis_title or y_column,
        autosize=config.autosize,
        width=config.width,
        height=config.height,
        margin=config.margin,
        font=dict(color=config.font_color)
    )

    return fig




class BasePlotting:
    def __init__(self, default_color='#3DAFA8'):
        self.default_color = default_color

    def create_histogram(self, data, x_column, title, xaxis_title, yaxis_title):
        """
        Create a histogram using Plotly with the specified default color for bars.
        """
        fig = px.histogram(data, x=x_column, nbins=30, marginal="box", title=title, template='plotly_white')
        fig.update_traces(marker=dict(line=dict(width=1.5, color="#FF7F7F"), color=self.default_color))
        fig.update_layout(
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            template='plotly_white'
        )
        return fig

    def create_kde_histogram(self, data, x_column, title, xaxis_title, yaxis_title, kde_color=None):
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
            go.Scatter(x=x_range, y=y_kde * len(data) * np.diff(x_range)[0], mode='lines',
                       name='KDE', line=dict(color=kde_plot_color))
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
            data=data.to_dict('records'),
            columns=[{"name": i, "id": i} for i in data.columns]
        ).render()
