import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from config.config_managers import (ColorMap, DecisionScatterConfig,
                                    MatrixConfig)
from config.enums import CorrelationColumns, HoverColumns, TrainColumns
from managers.tabs.tab_managers import (BaseAnalysis, BasePlotting,
                                        create_bar_chart,
                                        create_correlation_matrix_plot,
                                        create_scatter_plot_with_color,
                                        create_scatter_plot_with_color_selected,
                                        create_similarity_matrix_plot)


def create_confusion_table(errors, true_col, pred_col):
    confusion = errors.pivot_table(
        index=true_col, columns=pred_col, aggfunc="size", fill_value=0
    )
    return confusion


class CorrelationMatrix(BaseAnalysis):
    @BaseAnalysis.handle_errors
    def generate_matrix(self, selected_df, correlation_method, selected_columns=None):
        """
        Calculate and return correlation matrix and scatter plot for selected data.
        """
        if selected_columns:
            correlation_columns = selected_columns
        else:
            columns = CorrelationColumns()
            correlation_columns = columns.get_columns()
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
            # filtered = selected_df[selected_df != -1].copy()
            # import pandas as pd
            # filtered = selected_df.replace(-1, pd.NA)
            # filtered = filtered.dropna()
            # correlation_matrix = filtered[correlation_columns].corr(
            #     method=correlation_method
            # )
            correlation_matrix = selected_df[correlation_columns].corr(
                method=correlation_method
            )
            
            
            # # Create a mask for the upper triangle
            # mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

            # # Set the values in the upper triangle to NaN
            # correlation_matrix = correlation_matrix.mask(mask)
            config = MatrixConfig(
                title=f"Behavioural Metrics Correlation ({correlation_method.capitalize()} Coefficients)",
                x="Variables",
                y="Variables",
                color="Correlation",
                color_continuous_scale="RdBu_r",
                font_color="#000000",
                width=700,  # Slightly larger width for better visibility
                height=600,
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
            title=f"Training Representation Scatter Plot by {columns.TRUE_LABELS.value}",
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
        if x_column == "Pre X" and y_column == "Pre Y":
            title_prefix = "Pre-trained"
        else:
            title_prefix = "Fine-tuned"

        title = f"Representation Space Scatter Plot ({title_prefix} Model)"
        hover_columns = HoverColumns
        color_map = ColorMap()
        scatter_config = DecisionScatterConfig(
            title=title,
            hover_data=hover_columns.list_columns(),
            xaxis_title=x_column,
            yaxis_title=y_column,
            color_discrete_map=color_map.color_map,
        )
        return create_scatter_plot_with_color(
            selected_df, x_column, y_column, color_column, symbol_column, scatter_config
        )
        
    def generate_selection_highlighted_figure(
        self,
        df: pd.DataFrame,
        x_column: str,
        y_column: str,
        color_column: str,
        selected_ids: list,
        
    ):
        if x_column == "Pre X" and y_column == "Pre Y":
            title_prefix = "Pre-trained"
        else:
            title_prefix = "Fine-tuned"
        title = f"Representation Space Scatter Plot ({title_prefix} Model)"
        hover_columns = HoverColumns
        color_map = ColorMap()
        scatter_config = DecisionScatterConfig(
            title=title,
            hover_data=hover_columns.list_columns(),
            xaxis_title=x_column,
            yaxis_title=y_column,
            color_discrete_map=color_map.color_map,
        )
        labels = df[color_column].unique()
        
        fig = go.Figure()

        for label in labels:
            df_label = df[df[color_column] == label]
            selected_mask = df_label["Global Id"].isin(selected_ids)

            # Unselected points
            df_unselected = df_label[~selected_mask]
            fig.add_trace(
                go.Scattergl(
                    x=df_unselected[x_column],
                    y=df_unselected[y_column],
                    mode='markers',
                    name=str(label),
                    marker=dict(
                        color=color_map.color_map.get(label, 'gray'),
                        size=scatter_config.unselected_marker_size,
                        opacity=scatter_config.unselected_opacity
                    ),
                    text=df_label.loc[~selected_mask, scatter_config.hover_data].astype(str).agg("<br>".join, axis=1),
                    hoverinfo="text",
                    # text=df_unselected["Global Id"],
                )
            )

            # Selected points
            df_selected = df_label[selected_mask]
            if not df_selected.empty:
                fig.add_trace(
                    go.Scattergl(
                        x=df_selected[x_column],
                        y=df_selected[y_column],
                        mode='markers',
                        name=f"{label} (Selected)",
                        # marker=dict(
                        #     color=color_map.color_map.get(label, 'black'),
                        #      size=8,
                        #     opacity=0.5,
                        #     line=dict(width=1, color='black')
                        # ),
                        
                        marker=dict(
                            color=color_map.color_map.get(label, 'black'),  # Preserve original color
                            size=scatter_config.selected_marker_size,       # e.g., 8 or 10
                            opacity=scatter_config.selected_opacity,        # e.g., 1.0
                            symbol='diamond',                               # Change marker shape
                            line=dict(width=1.5, color='black')             # Outline for emphasis
                        ),
                        text=df_label.loc[selected_mask, scatter_config.hover_data].astype(str).agg("<br>".join, axis=1),
                        hoverinfo="text",
                    )
                )

        fig.update_layout(
            template="plotly_white",  # 🟢 WHITE BACKGROUND
            title=scatter_config.title,
            xaxis_title=scatter_config.xaxis_title or x_column,
            yaxis_title=scatter_config.yaxis_title or y_column,
            autosize=scatter_config.autosize,
            width=scatter_config.width,
            height=scatter_config.height,
            xaxis=dict(visible=scatter_config.xaxis_visible, showgrid=scatter_config.xaxis_showgrid),
            yaxis=dict(visible=scatter_config.yaxis_visible, showgrid=scatter_config.yaxis_showgrid),
            showlegend=True,
        )
        
        if not df_selected.empty:
            x0 = df_selected[x_column].min()
            x1 = df_selected[x_column].max()
            y0 = df_selected[y_column].min()
            y1 = df_selected[y_column].max()

            # fig.add_shape(
            #     type="rect",
            #     x0=x0, x1=x1,
            #     y0=y0, y1=y1,
            #     line=dict(color="rgba(0, 0, 0, 0.05)", dash="dot"),
            #     fillcolor="rgba(0, 0, 0, 0.05)",
            #     layer="below",
            # )

        return fig
        

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
            # title=f"Behavioural Metric Scatter Plot: {x_column} vs {y_column} (Coloured by {color_column})",
            title=f"Behavioural Metric Scatter Plot",
            hover_data=hover_columns.list_columns(),
            xaxis_title=x_column,
            yaxis_title=y_column,
            xaxis_visible=axis_visible,
            yaxis_visible=axis_visible,
            xaxis_showgrid=grid_visible,
            yaxis_showgrid=grid_visible,
            color_discrete_map=color_map.color_map,
        )
        # return create_scatter_plot_with_color(
        #     selected_df, x_column, y_column, color_column, symbol_column, scatter_config
        # )
        fig = create_scatter_plot_with_color(
            selected_df, x_column, y_column, color_column, symbol_column, scatter_config
        )
        non_spatial = (x_column not in ["X", "Pre X"]) and (y_column not in ["Y", "Pre Y"])
        if non_spatial:
            try:
                x_mean = selected_df[x_column].mean()
                y_mean = selected_df[y_column].mean()

                # Add dashed mean lines
                fig.add_vline(
                    x=x_mean,
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5,
                )
                fig.add_hline(
                    y=y_mean,
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5,
                )

                # # Add text labels for the mean values near the lines
                # fig.add_annotation(
                #     x=x_mean,
                #     y=selected_df[y_column].min(),  # Anchor at bottom to avoid points
                #     text=f"Mean X: {x_mean:.2f}",
                #     showarrow=False,
                #     yshift=-20,
                #     font=dict(size=12, color="gray"),
                #     textangle=90,
                # )

                # fig.add_annotation(
                #     x=selected_df[x_column].min(),  # Anchor at left
                #     y=y_mean,
                #     text=f"Mean Y: {y_mean:.2f}",
                #     showarrow=False,
                #     xshift=-20,
                #     font=dict(size=12, color="gray"),
                # )

            except Exception as e:
                logging.warning(f"[MeasurePlot] Could not add mean lines or labels: {e}")

        return fig
        
    def create_measure_scatter_highlighted(
        self,
        df: pd.DataFrame,
        x_column: str,
        y_column: str,
        color_column: str,
        selected_ids: list,
    ):
        hover_columns = HoverColumns
        axis_visible = not (x_column == "Pre X" and y_column == "Pre Y")
        grid_visible = axis_visible
        color_map = ColorMap()
        scatter_config = DecisionScatterConfig(
            # title=f"Behavioural Metric Scatter Plot: {x_column} vs {y_column} (Coloured by {color_column})",
            title=f"Behavioural Metric Scatter Plot",
            hover_data=hover_columns.list_columns(),
            xaxis_title=x_column,
            yaxis_title=y_column,
            xaxis_visible=axis_visible,
            yaxis_visible=axis_visible,
            xaxis_showgrid=grid_visible,
            yaxis_showgrid=grid_visible,
            color_discrete_map=color_map.color_map,
        )
        fig = go.Figure()
        labels = df[color_column].unique()

        for label in labels:
            df_label = df[df[color_column] == label]
            selected_mask = df_label["Global Id"].isin(selected_ids)

            # Unselected
            df_unselected = df_label[~selected_mask]
            fig.add_trace(go.Scattergl(
                x=df_unselected[x_column],
                y=df_unselected[y_column],
                mode='markers',
                name=str(label),
                marker=dict(
                    color=scatter_config.color_discrete_map.get(label, 'gray'),
                    size=scatter_config.unselected_marker_size,
                    opacity=scatter_config.unselected_opacity,
                ),
                text=df_unselected[scatter_config.hover_data].astype(str).agg("<br>".join, axis=1),
                hoverinfo="text",
            ))

            # Selected
            df_selected = df_label[selected_mask]
            if not df_selected.empty:
                fig.add_trace(go.Scattergl(
                    x=df_selected[x_column],
                    y=df_selected[y_column],
                    mode='markers',
                    name=f"{label} (Selected)",
                    marker=dict(
                            color=color_map.color_map.get(label, 'black'),  # Preserve original color
                            size=scatter_config.selected_marker_size,       # e.g., 8 or 10
                            opacity=scatter_config.selected_opacity,        # e.g., 1.0
                            symbol='diamond',                               # Change marker shape
                            line=dict(width=1.5, color='black')             # Outline for emphasis
                    ),
                    text=df_label.loc[selected_mask, scatter_config.hover_data].astype(str).agg("<br>".join, axis=1),
                    hoverinfo="text",
                ))
        
        
        if not df_selected.empty:
            x0 = df_selected[x_column].min()
            x1 = df_selected[x_column].max()
            y0 = df_selected[y_column].min()
            y1 = df_selected[y_column].max()

            # fig.add_shape(
            #     type="rect",
            #     x0=x0, x1=x1,
            #     y0=y0, y1=y1,
            #     line=dict(color="rgba(0, 0, 0, 0.05)", dash="dot"),
            #     fillcolor="rgba(0, 0, 0, 0.05)",
            #     layer="below",
            # )

        fig.update_layout(
            title=scatter_config.title,
            xaxis_title=scatter_config.xaxis_title,
            yaxis_title=scatter_config.yaxis_title,
            autosize=scatter_config.autosize,
            width=scatter_config.width,
            height=scatter_config.height,
            template=scatter_config.template,
            xaxis=dict(visible=scatter_config.xaxis_visible, showgrid=scatter_config.xaxis_showgrid),
            yaxis=dict(visible=scatter_config.yaxis_visible, showgrid=scatter_config.yaxis_showgrid),
            showlegend=True,
        )
        # 🔹 Add mean lines if relevant
        non_spatial = (x_column not in ["X", "Pre X"]) and (y_column not in ["Y", "Pre Y"])
        if non_spatial:
            try:
                x_mean = df[x_column].mean()
                y_mean = df[y_column].mean()

                fig.add_vline(x=x_mean, line_dash="dash", line_color="gray", opacity=0.5)
                fig.add_hline(y=y_mean, line_dash="dash", line_color="gray", opacity=0.5)
            except Exception as e:
                logging.warning(f"[MeasurePlot] Could not add mean lines: {e}")
        return fig


    
        
class SelectionTagProportion(BaseAnalysis):
    @BaseAnalysis.handle_errors
    def generate_plot(self, df, columns, category):
        
        x_column = category if category else columns.PRED_LABELS.value
        confusion_table = create_confusion_table(
            df, columns.TRUE_LABELS.value, x_column
        )
        # confusion_table = create_confusion_table(
        #     df, columns.TRUE_LABELS.value, columns.PRED_LABELS.value
        # )
        color_map = ColorMap()
        selection_tag_proportion = create_bar_chart(
            data=confusion_table,
            title=f"Selection Tag Proportion for {x_column}",
            # color_column=columns.PRED_LABELS.value,
            # xaxis_title=columns.PRED_LABELS.value,
            color_column=x_column,
            xaxis_title=x_column,
            yaxis_title=columns.TRUE_LABELS.value,
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
