import logging

import numpy as np
import pandas as pd

from config.enums import (CorrelationCoefficients, DecisionType,
                          DisplayColumns,
                          SelectionPlotColumns)
from managers.plotting.decision_plotting_managers import (
    CentroidAverageSimilarity, CorrelationMatrix, DecisionScatter,
    MeasureScatter, SelectionTagProportion, SimilarityMatrix, TrainScatter)
from managers.tabs.tab_managers import BaseTabManager


class DecisionTabManager(BaseTabManager):
    def __init__(self, variants_data):
        super().__init__(variants_data)
    
    def get_sentence_ids(self, variant, selected_ids=None):
        
        tab_data = self.get_tab_data(variant)
        if not tab_data or tab_data.analysis_data.empty:
            return None  # Let the callback handle PreventUpdate

        df = tab_data.analysis_data
        if 'Sentence Ids' not in df.columns:
            return None
        if selected_ids:
           df = df[df['Global Id'].isin(selected_ids)]

        return df['Sentence Ids'].unique().tolist()

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

    def get_filtered_analysis_data(self, variant, selected_column=None, selected_value=None):
        tab_data = self.get_tab_data(variant)

        if not tab_data or tab_data.analysis_data.empty:
            return None

        df = tab_data.analysis_data.copy()
        display_cols = DisplayColumns().get_columns()
        display_cols = [col for col in display_cols if col in df.columns]

        if selected_column and selected_value:
            try:
                values = selected_value.split() if isinstance(selected_value, str) else [selected_value]
                df = df[df[selected_column].isin(values)]
            except Exception:
                return None

        df = df[display_cols].round(3)
        df = df.rename(columns={"Global Id": "id"})  # ✅ renames in place
        return df


    def generate_matrix(self, variant, correlation_method, selected_columns=None):
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

        return correlation_analysis.generate_matrix(selected_df, coefficient, selected_columns)
    
    def generate_matrix_from_df(self, df, correlation_method, selected_columns=None):
        if df.empty:
            return None

        correlation_analysis = CorrelationMatrix()
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

        return correlation_analysis.generate_matrix(
            selected_df=df,
            correlation_method=coefficient,
            selected_columns=selected_columns,
        )

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
        color = color_column
        selected_df = tab_data.analysis_data
        if selected_df.empty:
            logging.error("No relevant data available after filtering.")
            return None
        if selection_ids:
            color = 'color'
            selected_df[color] = np.where(
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
            color_column=color,
            symbol_column=symbol_column,
        )

    def generate_measure_plot(
        self,
        variant,
        model_type,
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

        color = color_column
        selected_df = self.filter_ignored(tab_data.analysis_data)
        if selected_df.empty:
            logging.error("No relevant data available after filtering.")
            return None
        if selection_ids:
            color = 'color'
            selected_df[color] = np.where(
                selected_df["Global Id"].isin(selection_ids),
                "SELECTED",
                selected_df[color_column],
            )
            logging.info("Selected points are filtered and modified.")

        if not color_column:
            logging.error("Please select color column.")
        
        if model_type is not None:
            try:
                model_type_enum = DecisionType(model_type)
            except ValueError:
                logging.error("Invalid decision type selected.")
                return None

            if model_type_enum == DecisionType.FINETUNED:
                x_column = "X"
                y_column = "Y"
            elif model_type_enum == DecisionType.PRETRAINED:
                x_column = "Pre X"
                y_column = "Pre Y"
            else:
                logging.error("Unknown Model.")
                return None

        measure_analysis = MeasureScatter()

        return measure_analysis.generate_plot(
            selected_df,
            x_column=x_column,
            y_column=y_column,
            color_column=color,
            symbol_column=symbol_column,
        )
    
    def generate_measure_plot_from_ids(
        self,
        ids,
        variant,
        model_type,
        x_column,
        y_column,
        color_column,
        symbol_column=None,
        selection_ids=None,
    ):
        
        tab_data = self.get_tab_data(variant)
        if not tab_data:
            logging.error("No data available for the selected variant.")
            return None
        color = color_column
        selected_df = self.filter_ignored(tab_data.analysis_data)
        if selected_df.empty:
            logging.error("No relevant data available after filtering.")
            return None
        
        filtered_row_ids = [r for r in ids if r is not None]
        selected_df = selected_df[selected_df['Global Id'].isin(filtered_row_ids)]
        if selected_df.empty:
            logging.error("No relevant data available after filtering.")
            return None
        

        if selection_ids:
            color = "color"
            selected_df[color] = np.where(
                selected_df["Global Id"].isin(selection_ids),
                "SELECTED",
                selected_df[color_column],
            )

        if not color_column:
            logging.error("Please select a color column.")
            return None

        if model_type is not None:
            try:
                model_type_enum = DecisionType(model_type)
            except ValueError:
                logging.error("Invalid decision type selected.")
                return None

            if model_type_enum == DecisionType.FINETUNED:
                x_column = "X"
                y_column = "Y"
            elif model_type_enum == DecisionType.PRETRAINED:
                x_column = "Pre X"
                y_column = "Pre Y"
            else:
                logging.error("Unknown Model.")
                return None

        measure_analysis = MeasureScatter()

        return measure_analysis.generate_plot(
            selected_df,
            x_column=x_column,
            y_column=y_column,
            color_column=color,
            symbol_column=symbol_column,
        )


    # def generate_selection_data_table(
    #     self,
    #     variant,
    #     x_column,
    #     y_column,
    #     color_column,
    #     symbol_column=None,
    #     selection_ids=None,
    # ):
    #     """
    #     Calculate and return correlation matrix and scatter plot for selected data.
    #     """
    #     tab_data = self.get_tab_data(variant)

    #     if not tab_data:
    #         logging.error("No data available for the selected variant.")
    #         return None

    #     selected_df = self.filter_ignored(tab_data.analysis_data)
    #     if selected_df.empty:
    #         logging.error("No relevant data available after filtering.")
    #         return None
    #     if selection_ids:
    #         selected_df[color_column] = np.where(
    #             selected_df["Global Id"].isin(selection_ids),
    #             "SELECTED",
    #             selected_df[color_column],
    #         )
    #         logging.info("Selected points are filtered and modified.")

    #     if not color_column:
    #         logging.error("Please select color column.")

    #     measure_analysis = MeasureScatter()

    #     return measure_analysis.generate_plot(
    #         selected_df,
    #         x_column=x_column,
    #         y_column=y_column,
    #         color_column=color_column,
    #         symbol_column=symbol_column,
    #     )

    def generate_kmeans_results(self, variant):
        """
        Calculate and return centroid table.
        """
        tab_data = self.get_tab_data(variant)
        if not tab_data:
            logging.error("No data available for the selected variant.")
            return None

        selected_df = tab_data.kmeans_results
        
        if selected_df.empty:
            logging.error("No relevant data available after filtering.")
            return None
        
        return selected_df
    
    
    
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
    
    def generate_selection_summary(self, variant, category, selection_ids=None):
        """
        Calculate and return centroid table.
        """
        if category is None:
            category = 'Pred Labels'
        tab_data = self.get_tab_data(variant)
        plot_data = None
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

        if category not in selected_df.columns:
            logging.error(f"Column '{category}' not found in the dataset.")
            return None
        
        # Numeric summary (describe on metric columns)
        metric_columns = DisplayColumns().metric_columns
        numeric_df = plot_data[metric_columns].select_dtypes(include="number")
        summary_df = numeric_df.describe().T.round(2)
        summary_df.insert(0, "Metric", summary_df.index)  # 👈 Adds a visible column name
        summary_df.reset_index(drop=True, inplace=True)   # Optional: clean index

        return {
            "categorical_summary": DecisionTabManager.custom_summary(plot_data, category),
            "numeric_summary": summary_df
        }


    def generate_tag_proportion(self, variant, column, selection_ids=None):
        """
        Calculate and return correlation matrix and scatter plot for selected data.
        """
        tab_data = self.get_tab_data(variant)
        plot_data = None

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

        return selection_analysis.generate_plot(plot_data, selection_columns, column)


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
            tab_data.attention_similarity_matrix, "Attention Similarity Matrix"
        )
        attention_weights = similarity_analysis.generate_matrix(
            tab_data.attention_weights_similarity_matrix, "Attention Weights Similarity Matrix",
        )
        if attention_matrices is None and attention_weights is None:
            logging.error("No Training Impact Available.")
            return None, None
        return attention_matrices, attention_weights
    
    @staticmethod
    def custom_summary(df, col):
        counts = df[col].value_counts()
        percents = df[col].value_counts(normalize=True) * 100

        summary_df = pd.DataFrame({
            'Label': counts.index.astype(str),
            'Count': counts.values,
            'Percent': percents.map(lambda x: f"{x:.2f}%")
        })

        # Add total row
        total_count = counts.sum()
        summary_df.loc[len(summary_df.index)] = {
            'Label': 'Total',
            'Count': total_count,
            'Percent': '100.00%'
        }

        return summary_df
