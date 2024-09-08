import logging

import numpy as np

from config.enums import (CorrelationCoefficients, DecisionType,
                          SelectionPlotColumns)
from managers.plotting.decision_plotting_managers import (
    CentroidAverageSimilarity, CorrelationMatrix, DecisionScatter,
    MeasureScatter, SelectionTagProportion, SimilarityMatrix, TrainScatter)
from managers.tabs.tab_managers import BaseTabManager


class DecisionTabManager(BaseTabManager):
    def __init__(self, variants_data):
        super().__init__(variants_data)

    def get_training_data(self, variant):
        """Fetch training data for a specific variant."""
        tab_data = self.get_tab_data(variant)

        if not tab_data:
            logging.error("No data available for the selected variant.")
            return None
        try:
            data = tab_data.train_data
        except ValueError:
            logging.error("Invalid train data.")
            return None

        if data is None or data.empty:
            logging.warning("No training data available.")
            return None

        train_analysis = TrainScatter()

        return train_analysis.generate_plot(data)

    def generate_matrix(self, variant, correlation_method):
        """
        Calculate and return correlation matrix and scatter plot for selected data.
        """
        tab_data = self.get_tab_data(variant)

        if not tab_data:
            logging.error("No data available for the selected variant.")
            return None

        selected_df = self.filter_ignored(tab_data.analysis_data)
        if selected_df.empty:
            logging.error("No relevant data available after filtering.")
            return None

        try:
            coefficient_type_enum = CorrelationCoefficients(correlation_method)
        except ValueError:
            logging.error("Invalid coefficient selected.")
            return None

        if coefficient_type_enum == CorrelationCoefficients.PEARSON:
            coefficient = "pearson"
        elif coefficient_type_enum == CorrelationCoefficients.SPEARMAN:
            coefficient = "spearman"
        else:
            logging.error("Unknown coefficient.")
            return None

        correlation_analysis = CorrelationMatrix()

        return correlation_analysis.generate_matrix(selected_df, coefficient)

    def generate_decision_plot(
        self,
        variant,
        decision_type,
        color_column,
        symbol_column=None,
        selection_ids=None,
    ):
        """
        Calculate and return correlation matrix and scatter plot for selected data.
        """
        tab_data = self.get_tab_data(variant)

        if not tab_data:
            logging.error("No data available for the selected variant.")
            return None

        # selected_df = self.filter_ignored(tab_data.analysis_data)
        selected_df = tab_data.analysis_data
        if selected_df.empty:
            logging.error("No relevant data available after filtering.")
            return None
        if selection_ids:
            selected_df[color_column] = np.where(
                selected_df["Global Id"].isin(selection_ids),
                "SELECTED",
                selected_df[color_column],
            )
            logging.info("Selected points are filtered and modified.")
        try:
            decision_type_enum = DecisionType(decision_type)
        except ValueError:
            logging.error("Invalid decision type selected.")
            return None

        if decision_type_enum == DecisionType.FINETUNED:
            x_column = "X"
            y_column = "Y"
        elif decision_type_enum == DecisionType.PRETRAINED:
            x_column = "Pre X"
            y_column = "Pre Y"
        else:
            logging.error("Unknown Model.")
            return None

        if not color_column:
            logging.error("Please select color column.")

        decision_analysis = DecisionScatter()

        return decision_analysis.generate_plot(
            selected_df,
            x_column=x_column,
            y_column=y_column,
            color_column=color_column,
            symbol_column=symbol_column,
        )

    def generate_measure_plot(
        self,
        variant,
        x_column,
        y_column,
        color_column,
        symbol_column=None,
        selection_ids=None,
    ):
        """
        Calculate and return correlation matrix and scatter plot for selected data.
        """
        tab_data = self.get_tab_data(variant)

        if not tab_data:
            logging.error("No data available for the selected variant.")
            return None

        selected_df = self.filter_ignored(tab_data.analysis_data)
        if selected_df.empty:
            logging.error("No relevant data available after filtering.")
            return None
        if selection_ids:
            selected_df[color_column] = np.where(
                selected_df["Global Id"].isin(selection_ids),
                "SELECTED",
                selected_df[color_column],
            )
            logging.info("Selected points are filtered and modified.")

        if not color_column:
            logging.error("Please select color column.")

        measure_analysis = MeasureScatter()

        return measure_analysis.generate_plot(
            selected_df,
            x_column=x_column,
            y_column=y_column,
            color_column=color_column,
            symbol_column=symbol_column,
        )

    def generate_selection_data_table(
        self,
        variant,
        x_column,
        y_column,
        color_column,
        symbol_column=None,
        selection_ids=None,
    ):
        """
        Calculate and return correlation matrix and scatter plot for selected data.
        """
        tab_data = self.get_tab_data(variant)

        if not tab_data:
            logging.error("No data available for the selected variant.")
            return None

        selected_df = self.filter_ignored(tab_data.analysis_data)
        if selected_df.empty:
            logging.error("No relevant data available after filtering.")
            return None
        if selection_ids:
            selected_df[color_column] = np.where(
                selected_df["Global Id"].isin(selection_ids),
                "SELECTED",
                selected_df[color_column],
            )
            logging.info("Selected points are filtered and modified.")

        if not color_column:
            logging.error("Please select color column.")

        measure_analysis = MeasureScatter()

        return measure_analysis.generate_plot(
            selected_df,
            x_column=x_column,
            y_column=y_column,
            color_column=color_column,
            symbol_column=symbol_column,
        )

    def generate_tag_proportion(self, variant, selection_ids=None):
        """
        Calculate and return correlation matrix and scatter plot for selected data.
        """
        tab_data = self.get_tab_data(variant)

        if not tab_data:
            logging.error("No data available for the selected variant.")
            return None

        selected_df = self.filter_ignored(tab_data.analysis_data)
        if selected_df.empty:
            logging.error("No relevant data available after filtering.")
            return None
        if selection_ids:
            plot_data = selected_df[selected_df["Global Id"].isin(selection_ids)]
            logging.info("Selected points are filtered and modified.")

        selection_columns = SelectionPlotColumns
        selection_analysis = SelectionTagProportion()

        return selection_analysis.generate_plot(plot_data, selection_columns)

    def generate_centroid_matrix(self, variant):
        """
        Calculate and return centroid table.
        """
        tab_data = self.get_tab_data(variant)

        if not tab_data:
            logging.error("No data available for the selected variant.")
            return None

        selected_df = tab_data.centroids_avg_similarity_matrix
        if selected_df.empty:
            logging.error("No relevant data available after filtering.")
            return None
        centroid_similarity = CentroidAverageSimilarity()
        return centroid_similarity.generate_plot(selected_df)

    def generate_training_impact(self, variant):
        """
        Calculate and return centroid table.
        """
        tab_data = self.get_tab_data(variant)
        if not tab_data:
            logging.error("No data available for the selected variant.")
            return None

        similarity_analysis = SimilarityMatrix()

        attention_matrices = similarity_analysis.generate_matrix(
            tab_data.attention_similarity_matrix
        )
        attention_weights = similarity_analysis.generate_matrix(
            tab_data.attention_weights_similarity_matrix
        )
        if attention_matrices is None and attention_weights is None:
            logging.error("No Training Impact Available.")
            return None, None
        return attention_matrices, attention_weights
