import logging

import numpy as np
import pandas as pd
from dash import html

from config.config_managers import (BarConfig, ColorMap, MatrixConfig,
                                    ScatterConfig, ScatterWidthConfig,
                                    ViolinConfig)
from config.enums import (CorrelationColumns, CustomAnalysisConfig,
                          ErrorRateColumns, TagAmbiguityColumns,
                          TokenVariabilityColumns)
from layouts.managers.layout_managers import CustomDataTable
from managers.tabs.tab_managers import (BasePlotting,
                                        create_correlation_matrix_plot,
                                        create_histogram_plot,
                                        create_scatter_plot,
                                        create_scatter_width_plot,
                                        create_violin_plot)


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


class DistributionAnalysis(BaseAnalysis):
    @BaseAnalysis.handle_errors
    def generate_violin_plot(self, data, distribution_column, categorical_column):
        """Generate a violin plot with matching theme and style."""
        logging.debug(
            f"Generating violin plot for {distribution_column} by {categorical_column}"
        )

        # Configure the plot
        config = ViolinConfig(
            title=f"Violin Plot of {distribution_column} by {categorical_column}",
            line_color="#3DAFA8",
            box_line_color="#3DAFA8",
            meanline_color="#3DAFA8",
            marker_color="#FF7F7F",
            font_color="#000000",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
        )

        # Generate the plot
        violin_fig = create_violin_plot(
            data,
            distribution_column=distribution_column,
            categorical_column=categorical_column,
            config=config,
        )

        return violin_fig

    @BaseAnalysis.handle_errors
    def generate_distribution_plot(self, data, distribution_column, kde=True):
        """Generate a distribution plot with a KDE overlay using Plotly."""
        logging.debug(
            f"Generating distribution plot for {distribution_column} with KDE={kde}"
        )

        # Configure the plot
        config = BarConfig(
            title=f"Distribution of {distribution_column}",
            xaxis_title=distribution_column,
            line_color="#3DAFA8",
            kde_line_color="#FF7F7F",
            kde=True,
        )

        # Generate the plot
        distribution_fig = create_histogram_plot(data, distribution_column, config)

        return distribution_fig

    def generate_distribution_or_violin(
        self, selected_df, distribution_column, categorical_column
    ):
        """Generate a distribution or violin plot based on input parameters."""

        if selected_df is None or selected_df.empty:
            logging.error("No relevant data available after filtering.")
            return None

        if not distribution_column:
            return html.Span(
                "Please select a column and click 'Plot Distribution' to view plot."
            )

        figure = None
        if distribution_column and categorical_column:
            figure = self.generate_violin_plot(
                selected_df, distribution_column, categorical_column
            )
        else:
            figure = self.generate_distribution_plot(
                selected_df,
                distribution_column,
            )

        if figure:
            figure.update_layout(autosize=True)
            return figure

        return None


class TokenVariabilityAnalysis(BaseAnalysis):
    @BaseAnalysis.handle_errors
    def calculate_token_counts_and_ratios(self, df, columns):
        counts = df[columns.TRUE_LABELS.value].value_counts().sort_index()
        types = df.groupby(columns.TRUE_LABELS.value)[
            columns.CORE_TOKENS.value
        ].nunique()
        ratios = types / counts

        token_distribution_df = pd.DataFrame(
            {
                columns.RAW_COUNTS.value: counts,
                columns.TYPES.value: types,
                columns.COUNT_TYPE_RATIO.value: ratios,
            }
        )

        return token_distribution_df

    @BaseAnalysis.handle_errors
    def calculate_totals_and_ratios(self, df, token_distribution_df, columns):
        totals = df[columns.CORE_TOKENS.value].agg(["size", "nunique"]).tolist()
        ne_totals = (
            df[df[columns.TRUE_LABELS.value] != "O"][columns.CORE_TOKENS.value]
            .agg(["size", "nunique"])
            .tolist()
        )

        token_distribution_df.loc["Total"] = totals + [totals[1] / totals[0]]
        token_distribution_df.loc["Total NEs"] = ne_totals + [
            ne_totals[1] / ne_totals[0]
        ]

        token_distribution_df = token_distribution_df.rename(
            columns=columns.rename_mapping()
        )

        token_distribution_df[columns.NUMBER_OF_TOKENS.value] = token_distribution_df[
            columns.NUMBER_OF_TOKENS.value
        ].astype(int)
        token_distribution_df[columns.NUMBER_OF_TYPES.value] = token_distribution_df[
            columns.NUMBER_OF_TYPES.value
        ].astype(int)
        token_distribution_df[columns.TOKEN_TYPE_RATIO.value] = token_distribution_df[
            columns.TOKEN_TYPE_RATIO.value
        ].apply(lambda x: round(x, 3))

        token_distribution_df = token_distribution_df.sort_values(
            by=columns.NUMBER_OF_TOKENS.value, ascending=False
        )
        token_distribution_df = token_distribution_df.reset_index().rename(
            columns={columns.TRUE_LABELS.value: columns.CATEGORY.value}
        )

        token_distribution_df[columns.NEs_PROPORTION.value] = (
            token_distribution_df[columns.NUMBER_OF_TOKENS.value] / ne_totals[0]
        )
        token_distribution_df[columns.NEs_PROPORTION.value] = token_distribution_df[
            columns.NEs_PROPORTION.value
        ].apply(lambda x: round(x * 100, 2))

        return token_distribution_df

    def create_token_variability_table(self, selected_df):
        columns = TokenVariabilityColumns

        token_distribution_df = self.calculate_token_counts_and_ratios(
            selected_df, columns
        )
        if token_distribution_df is None:
            return None

        token_distribution_df = self.calculate_totals_and_ratios(
            selected_df, token_distribution_df, columns
        )
        if token_distribution_df is None:
            return None

        return self.render_table(token_distribution_df, "token_distribution_table")


class TagAmbiguityAnalysis(BaseAnalysis):
    @BaseAnalysis.handle_errors
    def aggregate_tag_ambiguity(self, df, columns):
        tag_ambiguity_analysis = (
            df.groupby([columns.TRUE_LABELS.value, columns.CORE_TOKENS.value])
            .agg(
                mean_consistency=(columns.CONSISTENCY.value, "mean"),
                mean_inconsistency=(columns.INCONSISTENCY.value, "mean"),
                mean_token_entropy=(columns.TOKEN_ENTROPY.value, "mean"),
                mean_dataset_token_entropy=(
                    columns.DATASET_TOKEN_ENTROPY.value,
                    "mean",
                ),
                mean_word_entropy=(columns.WORD_ENTROPY.value, "mean"),
                mean_dataset_word_entropy=(columns.DATASET_WORD_ENTROPY.value, "mean"),
            )
            .reset_index()
        )
        return tag_ambiguity_analysis

    @BaseAnalysis.handle_errors
    def summarize_tag_ambiguity(self, df, columns):

        summary = (
            df.groupby(columns.TRUE_LABELS.value)
            .agg(
                **{
                    agg_name: (col_name, "mean")
                    for agg_name, col_name in columns.aggregation_mapping().items()
                }
            )
            .reset_index()
            .round(3)
        )
        return summary

    def create_tag_ambiguity_table(self, selected_df):
        columns = TagAmbiguityColumns

        tag_ambiguity_analysis = self.aggregate_tag_ambiguity(selected_df, columns)
        if tag_ambiguity_analysis is None:
            return None

        tag_ambiguity_summary = self.summarize_tag_ambiguity(
            tag_ambiguity_analysis, columns
        )
        if tag_ambiguity_summary is None or tag_ambiguity_summary.empty:
            return None

        tag_ambiguity_summary = tag_ambiguity_summary.rename(
            columns=columns.rename_view_mapping()
        )
        return self.render_table(tag_ambiguity_summary, "tag_ambiguity_table")


class TokenLengthPlot(BasePlotting):
    def generate_plot(self, selected_df):
        """
        Generate the token length distribution plot.
        """
        config = CustomAnalysisConfig.get_plot_config("TOKEN_LENGTH")
        if not config:
            return None

        # Calculate token lengths
        selected_df.loc[:, config["x_column"]] = selected_df[config["column"]].apply(
            lambda x: len(str(x))
        )

        return self.create_kde_histogram(
            data=selected_df,
            x_column=config["x_column"],
            title=config["title"],
            xaxis_title=config["xaxis_title"],
            yaxis_title=config["yaxis_title"],
        )


class SentenceLengthPlot(BasePlotting):
    def generate_plot(self, selected_df):
        """
        Generate the sentence length distribution plot.
        """
        config = CustomAnalysisConfig.get_plot_config("SENTENCE_LENGTH")
        if not config:
            return None

        # Group by 'Sentence Id' and count the number of tokens per sentence
        sentence_length_df = (
            selected_df.groupby(config["groupby_column"])[config["column"]]
            .count()
            .reset_index()
        )
        sentence_length_df.columns = [config["groupby_column"], config["x_column"]]
        # Create the histogram
        return self.create_histogram(
            data=sentence_length_df,
            x_column=config["x_column"],
            title=config["title"],
            xaxis_title=config["xaxis_title"],
            yaxis_title=config["yaxis_title"],
        )


class ErrorRateAnalysis(BaseAnalysis):
    @BaseAnalysis.handle_errors
    def generate_weighted_error_rate_plot(self, df):
        """Calculate and generate a weighted error rate plot."""
        columns = ErrorRateColumns

        # Use the enum for column names
        true_labels = columns.TRUE_LABELS.value
        pred_labels = columns.PRED_LABELS.value
        core_tokens = columns.CORE_TOKENS.value
        tokenization_rate = columns.TOKENIZATION_RATE.value
        errors_col = columns.ERRORS.value
        total_count = columns.TOTAL_COUNT.value
        error_count = columns.ERROR_COUNT.value
        error_rate = columns.ERROR_RATE.value
        weighted_error_rate = columns.WEIGHTED_ERROR_RATE.value

        # Step 1: Calculate Error Rate
        df[errors_col] = df[true_labels] != df[pred_labels]
        errors = df[df[errors_col]]

        group_counts = df.groupby(tokenization_rate)[core_tokens].count().reset_index()
        group_counts.columns = [tokenization_rate, total_count]

        error_counts = (
            errors.groupby(tokenization_rate)[core_tokens].count().reset_index()
        )
        error_counts.columns = [tokenization_rate, error_count]

        # Merge counts and calculate the error rate
        error_rate_df = pd.merge(
            group_counts, error_counts, on=tokenization_rate, how="left"
        ).fillna(0)
        error_rate_df[error_rate] = (
            error_rate_df[error_count] / error_rate_df[total_count]
        )

        # Step 2: Calculate Weighted Error Rate
        total_words = df[core_tokens].count()
        error_rate_df[weighted_error_rate] = error_rate_df[error_rate] * (
            error_rate_df[total_count] / total_words
        )

        config = ScatterConfig(
            title=f"Weighted Error Rate by {tokenization_rate}",
            xaxis_title=tokenization_rate,
            yaxis_title=weighted_error_rate,
            line_color="#FF7F7F",
            marker_color="#3DAFA8",
            hover_data=[total_count, error_count],
        )

        # Generate the plot
        error_rate_fig = create_scatter_plot(
            error_rate_df, tokenization_rate, weighted_error_rate, config
        )

        return error_rate_fig


class CorrelationAnalysis(BaseAnalysis):
    @BaseAnalysis.handle_errors
    def calculate_correlation(
        self, selected_df, correlation_method, x_column, y_column
    ):
        """
        Calculate and return correlation matrix and scatter plot for selected data.
        """
        columns = CorrelationColumns
        true_labels_col = CorrelationColumns.TRUE_LABELS.value

        try:
            # Aggregate the DataFrame by true_labels and compute mean of each column
            aggregated_df = (
                selected_df.groupby(true_labels_col)[columns.list_columns()[1:]]
                .agg("mean")
                .reset_index()
            )
            numeric_cols = aggregated_df.select_dtypes(
                include=[np.number]
            ).columns.tolist()

            # Compute the correlation matrix

            correlation_matrix = aggregated_df[numeric_cols].corr(
                method=correlation_method
            )
            # Create a mask for the upper triangle
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

            # Set the values in the upper triangle to NaN
            correlation_matrix = correlation_matrix.mask(mask)
            config = MatrixConfig(
                title=f"{correlation_method.capitalize()} Correlation Matrix of Aggregated Data",
                x="Variables",
                y="Variables",
                color="Correlation",
                color_continuous_scale="RdBu_r",
                font_color="#000000",
                width=700,  # Slightly larger width for better visibility
                height=700,
            )

            # Generate the plot
            matrix_fig = create_correlation_matrix_plot(correlation_matrix, config)
            color_map = ColorMap()
            scatter_config = ScatterWidthConfig(
                title=f"Scatter Plot of {x_column} vs {y_column} by Label",
                xaxis_title=x_column,
                yaxis_title=y_column,
                line_color="#3DAFA8",
                marker_color=true_labels_col,
                width=700,
                height=700,
                color_discrete_map=color_map.color_map,
            )

            scatter_fig = create_scatter_width_plot(
                aggregated_df, x_column, y_column, scatter_config
            )

            return matrix_fig, scatter_fig

        except Exception as e:
            logging.error(f"Error in calculating correlation: {e}")
            return None, None
