import logging

import numpy as np

from config.config_managers import (ColorMap, DecisionScatterConfig,
                                    MatrixConfig)
from config.enums import CorrelationColumns, HoverColumns, TrainColumns
from managers.tabs.tab_managers import (BaseAnalysis, BasePlotting,
                                        create_bar_chart,
                                        create_correlation_matrix_plot,
                                        create_scatter_plot_with_color,
                                        create_similarity_matrix_plot)


def create_confusion_table(errors, true_col, pred_col):
    confusion = errors.pivot_table(
        index=true_col, columns=pred_col, aggfunc="size", fill_value=0
    )
    return confusion


class CorrelationMatrix(BaseAnalysis):
    @BaseAnalysis.handle_errors
    def generate_matrix(self, selected_df, correlation_method):
        """
        Calculate and return correlation matrix and scatter plot for selected data.
        """
        columns = CorrelationColumns()
        correlation_columns = columns.get_columns(include_confidence=True)

        try:
            # # Aggregate the DataFrame by true_labels and compute mean of each column
            # aggregated_df = (
            #     selected_df.groupby(true_labels_col)[columns.list_columns()]
            #     .agg("mean")
            #     .reset_index()
            # )
            # numeric_cols = selected_df.select_dtypes(
            #     include=[np.number]
            # ).columns.tolist()

            # Compute the correlation matrix

            correlation_matrix = selected_df[correlation_columns].corr(
                method=correlation_method
            )
            
            
            # # Create a mask for the upper triangle
            # mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

            # # Set the values in the upper triangle to NaN
            # correlation_matrix = correlation_matrix.mask(mask)
            config = MatrixConfig(
                title=f"{correlation_method.capitalize()} Correlation Matrix of Aggregated Data",
                x="Variables",
                y="Variables",
                color="Correlation",
                color_continuous_scale="RdBu_r",
                font_color="#000000",
                width=900,  # Slightly larger width for better visibility
                height=700,
            )

            # Generate the plot
            matrix_fig = create_correlation_matrix_plot(correlation_matrix, config)

            return matrix_fig

        except Exception as e:
            logging.error(f"Error in calculating correlation: {e}")

            return None


class TrainScatter(BasePlotting):
    def generate_plot(
        self,
        selected_df,
    ):
        """
        Generate the token length distribution plot.
        """
        columns = TrainColumns
        color_map = ColorMap()
        scatter_config = DecisionScatterConfig(
            title=f"Training Decision Boundary Scatter Plot by {columns.TRUE_LABELS.value}",
            hover_data=columns.list_columns(),
            xaxis_title=columns.X_COLUMN.value,
            yaxis_title=columns.Y_COLUMN.value,
            color_discrete_map=color_map.color_map,
        )
        return create_scatter_plot_with_color(
            selected_df,
            columns.X_COLUMN.value,
            columns.Y_COLUMN.value,
            columns.TRUE_LABELS.value,
            None,
            scatter_config,
        )


class DecisionScatter(BasePlotting):
    def generate_plot(
        self, selected_df, x_column, y_column, color_column, symbol_column
    ):
        """
        Generate the token length distribution plot.
        """
        hover_columns = HoverColumns
        color_map = ColorMap()
        scatter_config = DecisionScatterConfig(
            title=f"Decision Boundary Scatter Plot by {color_column}",
            hover_data=hover_columns.list_columns(),
            xaxis_title=x_column,
            yaxis_title=y_column,
            color_discrete_map=color_map.color_map,
        )
        return create_scatter_plot_with_color(
            selected_df, x_column, y_column, color_column, symbol_column, scatter_config
        )


class MeasureScatter(BasePlotting):
    def generate_plot(
        self, selected_df, x_column, y_column, color_column, symbol_column
    ):
        """
        Generate the token length distribution plot.
        """
        hover_columns = HoverColumns
        axis_visible = not (x_column == "Pre X" and y_column == "Pre Y")
        grid_visible = axis_visible
        color_map = ColorMap()
        scatter_config = DecisionScatterConfig(
            title=f"Measures Scatter Plot by {color_column}",
            hover_data=hover_columns.list_columns(),
            xaxis_title=x_column,
            yaxis_title=y_column,
            xaxis_visible=axis_visible,
            yaxis_visible=axis_visible,
            xaxis_showgrid=grid_visible,
            yaxis_showgrid=grid_visible,
            color_discrete_map=color_map.color_map,
        )
        return create_scatter_plot_with_color(
            selected_df, x_column, y_column, color_column, symbol_column, scatter_config
        )


class SelectionTagProportion(BaseAnalysis):
    @BaseAnalysis.handle_errors
    def generate_plot(self, df, columns):

        confusion_table = create_confusion_table(
            df, columns.TRUE_LABELS.value, columns.PRED_LABELS.value
        )
        color_map = ColorMap()
        selection_tag_proportion = create_bar_chart(
            data=confusion_table,
            title="Selection Tag Proportion",
            color_column=columns.PRED_LABELS.value,
            xaxis_title=columns.TRUE_LABELS.value,
            yaxis_title=columns.PRED_LABELS.value,
            width=700,
            height=700,
            color_map=color_map.color_map,
        )
        return selection_tag_proportion.update_layout(
            margin=dict(l=20, r=20, t=50, b=20)
        )


class CentroidAverageSimilarity(BaseAnalysis):
    def generate_plot(self, selected_df):
        config = MatrixConfig(
            title="Centroid Average Similarity Matrix",
            x="Centroids",
            y="NER_Label",
            color="Similarity",
            color_continuous_scale="RdBu_r",
            font_color="#000000",
            width=700,  # Slightly larger width for better visibility
            height=700,
        )
        plot_data = selected_df.set_index(config.y)
        # Generate the plot
        matrix_fig = create_similarity_matrix_plot(plot_data, config)
        return matrix_fig


class SimilarityMatrix(BaseAnalysis):
    @BaseAnalysis.handle_errors
    def generate_matrix(self, similarity_matrix, title):
        """
        Calculate and return correlation matrix and scatter plot for selected data.
        """

        config = MatrixConfig(
            title=title,
            x="Heads",
            y="Layers",
            color="Similarity Score",
            color_continuous_scale="RdBu_r",
            font_color="#000000",
            width=700,  # Slightly larger width for better visibility
            height=600,
        )

        # Generate the plot
        matrix_fig = create_correlation_matrix_plot(similarity_matrix, config)

        return matrix_fig
