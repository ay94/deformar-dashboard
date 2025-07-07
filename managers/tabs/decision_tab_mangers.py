import logging
import re
from config.config_managers import ColorMap
import numpy as np
import pandas as pd
from dash import html
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
    
    def get_training_sentence_ids(self, variant, selected_ids=None):
        
        tab_data = self.get_tab_data(variant)
        if not tab_data or tab_data.train_data.empty:
            return None  # Let the callback handle PreventUpdate

        df = tab_data.train_data
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

        return train_analysis.generate_plot(data), data['Sentence Ids'].unique()

    def get_filtered_analysis_data(self, variant, selected_column=None, selected_value=None):
        print(' O am selected values', selected_value)
        tab_data = self.get_tab_data(variant)

        if not tab_data or tab_data.analysis_data.empty:
            return None

        df = tab_data.analysis_data.copy()
        display_cols = DisplayColumns().get_columns()
        display_cols = [col for col in display_cols if col in df.columns]

        if selected_column and selected_value:
            try:
                values = [selected_value] if isinstance(selected_value, str) else selected_value
                
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
        
        if selection_ids:
            # color = 'color'
            # selected_df[color] = np.where(
            #     selected_df["Global Id"].isin(selection_ids),
            #     "SELECTED",
            #     selected_df[color_column],
            # )
            # logging.info("Selected points are filtered and modified.")
            return decision_analysis.generate_selection_highlighted_figure(
                df=selected_df,
                x_column=x_column,
                y_column=y_column,
                color_column=color,  # or whatever color category you're using
                selected_ids=selection_ids,
            )

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
        if selection_ids:
            # color = 'color'
            # selected_df[color] = np.where(
            #     selected_df["Global Id"].isin(selection_ids),
            #     "SELECTED",
            #     selected_df[color_column],
            # )
            # logging.info("Selected points are filtered and modified.")
            return measure_analysis.create_measure_scatter_highlighted(
                df=selected_df,
                x_column=x_column,
                y_column=y_column,
                color_column=color,  # or whatever color category you're using
                selected_ids=selection_ids,
            )

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
            
        if not color_column:
            logging.error("Please select a color column.")
            return None

        measure_analysis = MeasureScatter()
        if selection_ids:
            # color = "color"
            # selected_df[color] = np.where(
            #     selected_df["Global Id"].isin(selection_ids),
            #     "SELECTED",
            #     selected_df[color_column],
            # )
            
 
            return measure_analysis.create_measure_scatter_highlighted(
                    df=selected_df,
                    x_column=x_column,
                    y_column=y_column,
                    color_column=color,
                    selected_ids=selection_ids,
                )

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

    def render_training_entity_tags(self, entities, words, error_dict=None):
        if not entities:
            return [html.Span("No Errors", style={
                "color": ColorMap().color_map["No Errors"],
                "fontWeight": "bold"
            })]

        error_dict = error_dict or {}
        fp_set = set(error_dict.get("FP", []))
        fn_set = set(error_dict.get("FN", []))

        color_util = ColorMap()
        tags = []
        # Detect Arabic for layout direction
        is_arabic = any(re.search(r'[\u0600-\u06FF]', word) for word in words)
        direction = "rtl" if is_arabic else "ltr"
        text_align = "right" if is_arabic else "left"
        for ent in entities:
            _, etype, start, end = ent
            label = f"{etype} [{' '.join(words[start:end+1])}]"

            # Determine base color
            base_color = color_util.color_map.get(etype, "#ccc")

            # Override for errors
            if ent in fp_set:
                background = color_util.color_map.get("FP", "#EF553B")
                border = f"2px solid {background}"
            elif ent in fn_set:
                background = color_util.color_map.get("FN", "#00CC96")
                border = f"2px dashed {background}"
            else:
                background = base_color
                border = "2px solid transparent"

            style = {
                "backgroundColor": background,
                "color": "#fff",
                "padding": "6px 10px",
                "margin": "6px",
                "borderRadius": "6px",
                "display": "inline-block",
                "fontWeight": "bold",
                "fontSize": "0.85rem",
                "border": border,
                "direction": direction,
                "textAlign": text_align
            }

            tags.append(html.Span(label, style=style))

        return tags
    
    def get_sentence_slice(self, variant, instance_id):
        df = self.get_tab_data(variant).train_data
        sentence_df = df[df["Sentence Ids"] == instance_id]
        # Extract tokens and labels
        words_df = sentence_df[sentence_df['Labels'] != -100]
        words = words_df['Words']
        word_true_labels = words_df['True Labels']
        tokens = sentence_df["Tokens"].tolist()
        true_labels = sentence_df["True Labels"].tolist()
    
    
        return words, word_true_labels, tokens, true_labels
    
    def generate_training_output(self, variant, instance_id):
        words, word_true_labels, tokens, true_labels = self.get_sentence_slice(variant, instance_id)
        color_util = ColorMap()
        # ➕ Detect if Arabic (for RTL support)
        is_arabic = any(re.search(r'[\u0600-\u06FF]', word) for word in words)
        direction = "rtl" if is_arabic else "ltr"
        text_align = "right" if is_arabic else "left"
        
        sentence_colored = DecisionTabManager.generate_colored_tokens(words, word_true_labels, color_util.color_map)
        truth_colored = DecisionTabManager.generate_colored_tokens(tokens, true_labels, color_util.color_map)
        
        # ➕ Wrap in a styled Div that uses the correct direction
        wrapper_style = {
            "direction": direction,
            "textAlign": text_align,
            "lineHeight": "2em",
            "padding": "8px"
        }

        return (
            html.Div(sentence_colored, style=wrapper_style),
            html.Div(truth_colored, style=wrapper_style),
        )
    
   
    def get_training_entity_level_annotations_non_strict(self, variant, sentence_id):
        tab_data = self.get_tab_data(variant)
        if not tab_data or tab_data.train_data.empty:
            return [], [], {}, []

        df = tab_data.train_data
        core_data = df[df["Labels"] != -100]
        sentence_df = core_data[core_data["Sentence Ids"] == sentence_id]

        if sentence_df.empty:
            return [], [], {}, []

        y_true = sentence_df["True Labels"].tolist()
        words = sentence_df["Words"].tolist()

        analyzer = NonStrictEntityAnalyzer(sentence_df)
        true_entities = analyzer.extract_entities([y_true])[0]
        
        

        return true_entities, words
    
    
    def get_entity_level_annotations_strict(self, variant, sentence_id):
        tab_data = self.get_tab_data(variant)
        if not tab_data or tab_data.train_data.empty:
            return [], [], {}, []

        df = tab_data.train_data
        core_data = df[df["Labels"] != -100]
        sentence_df = core_data[core_data["Sentence Ids"] == sentence_id]

        if sentence_df.empty:
            return [], [], {}, []

        y_true = sentence_df["True Labels"].tolist()
        words = sentence_df["Words"].tolist()

        analyzer = StrictEntityAnalyzer(sentence_df)
        true_entities = analyzer.extract_entities([y_true]).entities[0]
        
        

        return true_entities, words
    
    @staticmethod
    def generate_colored_tokens(tokens, labels, color_map, show_label=True):
        """
        Generate a list of html.Span components with background color based on labels.

        Args:
            tokens (list[str]): The tokenized words.
            labels (list[str]): Corresponding labels for the tokens.
            color_map (dict): Label to color hex mapping.
            show_label (bool): Whether to show the label text on the token.

        Returns:
            list[html.Span]: Dash components for visual display.
        """
        styled_tokens = []
        for token, label in zip(tokens, labels):
            
            if label=='TN':
                continue  # 🚫 Skip True Negatives entirely
            if token in ['[CLS]', '[SEP]']:
                continue  # 🚫 Skip True Negatives entirely
            background_color = color_map.get(label, "#E0E0E0")
            token_display = f"{token} ({label})" if show_label and label != "O" else token

            styled_tokens.append(
                html.Span(
                    token_display,
                    style={
                        "backgroundColor": background_color,
                        "borderRadius": "6px",
                        "padding": "4px 6px",
                        "margin": "2px",
                        "display": "inline-block",
                        "color": "#fff" if label != "O" else "#000",
                        "fontWeight": "bold",
                    }
                )
            )
        return styled_tokens




from dataclasses import dataclass, field
from typing import List, Dict, Set
from abc import ABC, abstractmethod
from collections import defaultdict
from seqeval.scheme import Entities, IOB2, IOB1
from seqeval.metrics.sequence_labeling import get_entities
import pandas as pd
pd.set_option("display.max_rows", 200)  # Display all rows

class EntityErrorAnalyzer(ABC):
    """Abstract base class for entity analysis."""

    def __init__(self, df):
        self.df = df
        self.y_true = self.prepare_data(df)
        self.true_entities = []
        self.pred_entities = []

    @abstractmethod
    def extract_entities(self, y_data):
        """Extract entities based on the specific mode (strict or non-strict)."""
        pass

    @abstractmethod
    def prepare_entities(self):
        """Prepare true and predicted entities for analysis."""
        pass
    
    def prepare_data(self, df):
        core_data = df[df['Labels'] !=-100]
        y_true = core_data.groupby('Sentence Ids')['True Labels'].apply(list).tolist()
        
        return y_true
    
    def compute_false_negatives(self, entity_type):
        """Compute false negatives for a specific entity type."""
        return set(
            [e for e in self.true_entities if e[1] == entity_type]
        ) - set([e for e in self.pred_entities if e[1] == entity_type])

    def compute_false_positives(self, entity_type):
        """Compute false positives for a specific entity type."""
        return set(
            [e for e in self.pred_entities if e[1] == entity_type]
        ) - set([e for e in self.true_entities if e[1] == entity_type])

    def analyze_sentence_errors(self, target_entities, comparison_entities):
        """Analyze errors and return sentence IDs by error type."""
        error_sentences = defaultdict(set)  # Dictionary to hold sentence IDs for each error type
        non_o_errors = set()
        indexed_entities = defaultdict(list)

        # Index comparison entities by sentence
        for entity in comparison_entities:
            sen, entity_type, start, end = entity
            indexed_entities[sen].append(entity)

        # First pass: entity errors
        for target_entity in target_entities:
            t_sen, t_type, t_start, t_end = target_entity

            for comp_entity in indexed_entities[t_sen]:
                c_type, c_start, c_end = comp_entity[1:]

                if (
                    t_start == c_start
                    and t_end == c_end
                    and t_type != c_type
                    and target_entity not in non_o_errors
                ):
                    non_o_errors.add(target_entity)
                    error_sentences["Entity"].add(target_entity)

        # Second pass: boundary errors
        for target_entity in target_entities - non_o_errors:
            t_sen, t_type, t_start, t_end = target_entity

            for comp_entity in indexed_entities[t_sen]:
                c_sen, c_type, c_start, c_end = comp_entity

                if (
                    t_type == c_type
                    and (t_start <= c_start <= t_end or t_start <= c_end <= t_end)
                    and target_entity not in non_o_errors
                ):
                    non_o_errors.add(target_entity)
                    error_sentences["Boundary"].add(target_entity)

        # Third pass: combined entity and boundary errors
        for target_entity in target_entities - non_o_errors:
            t_sen, t_type, t_start, t_end = target_entity

            for comp_entity in indexed_entities[t_sen]:
                c_sen, c_type, c_start, c_end = comp_entity

                if (
                    c_type != t_type
                    and (t_start <= c_start <= t_end or t_start <= c_end <= t_end)
                    and target_entity not in non_o_errors
                ):
                    non_o_errors.add(target_entity)
                    error_sentences["Entity and Boundary"].add(target_entity)
                    # print(t_sen, t_start, t_end, c_sen, c_start, c_end)
                    # print(f' ({t_start} <= {c_start} <= {t_end} or {t_start} <= {c_end} <= {t_end})')
                    

        # Remaining unmatched errors are "O errors"
        for target_entity in target_entities - non_o_errors:
            t_sen, t_type, t_start, t_end = target_entity
            error_sentences["O"].add(target_entity)

        return {error_type: list(s_ids) for error_type, s_ids in error_sentences.items()}


    def analyze_component(self, error_type, entity_type=None):
        
        """Analyze errors (FP or FN) for a specific or all entity types."""
        self.prepare_entities()
        error_analysis = {}
        entity_types = (
            [entity_type]
            if entity_type
            else set(e[1] for e in self.true_entities + self.pred_entities)
        )

        for etype in entity_types:
            if error_type == "false_negatives":
                target_entities = self.compute_false_negatives(etype)
            elif error_type == "false_positives":
                target_entities = self.compute_false_positives(etype)
            else:
                raise ValueError("Error type must be 'false_negative' or 'false_positive'.")

            error_analysis[etype] = self.analyze_sentence_errors(
                target_entities, self.pred_entities if error_type == "false_negatives" else self.true_entities
            )

        return error_analysis
    
    def analyze_errors(self):
        self.prepare_entities()
        """Analyze both false positives and false negatives."""
        error_components = {"false_positives": defaultdict(set), "false_negatives": defaultdict(set)}

        for error_component in error_components.keys():
            results = self.analyze_component(error_component)
            for entity_type, errors in results.items():
                for error_type, sentences in errors.items():
                    error_components[error_component][error_type].update(sentences)

        # Convert sets to lists for consistency
        return {k: {etype: set(ids) for etype, ids in v.items()} for k, v in error_components.items()}
    
    


class StrictEntityAnalyzer(EntityErrorAnalyzer):
    """Analyzer for strict entity processing."""

    def extract_entities(self, y_data):
        """Extract entities in strict mode."""
        entities = Entities(y_data, IOB2, False)
        return self.adjust_end_index(entities)

    def prepare_entities(self):
        """Prepare true and predicted entities for strict mode."""
        self.true_entities = self.flatten_entities(self.extract_entities(self.y_true))
        self.pred_entities = self.flatten_entities(self.extract_entities(self.y_pred))

    def print_sentence(self, sen_id):
        """Print entities for a specific sentence ID."""
        true_entities = self.extract_entities(self.y_true).entities
        pred_entities = self.extract_entities(self.y_pred).entities
        print(f"True: {true_entities[sen_id]}")
        print(f"Pred: {pred_entities[sen_id]}")
        error = set(pred_entities[sen_id]) - set(true_entities[sen_id])
        print(f"Error in Pred: {error}")
        core_data = self.df[self.df['Labels'] !=-100]
        sentence_data = core_data[core_data['Sentence Ids']  == sen_id].copy()
        print(sentence_data[['Words', 'Sentence Ids', 'True Labels', 'Pred Labels', 'Strict True Entities', 'Strict Pred Entities', 'True Entities', 'Pred Entities']].head(60).to_string())

    @staticmethod
    def flatten_entities(entities):
        """Flatten strict entities into tuples."""
        return [e for sen in entities.entities for e in sen]
    
    @staticmethod
    def adjust_end_index(entities):
        """Adjust the end index for IOB2 entities to make them inclusive."""
        adjusted_entities = []
        for sentence_entities in entities.entities:  # Iterate through sentences
            adjusted_sentence = []
            for entity in sentence_entities:  # Iterate through entities in each sentence
                sentence_id, entity_type, start, end = entity.to_tuple()
                # Adjust end index
                adjusted_sentence.append((sentence_id, entity_type, start, end - 1))
            adjusted_entities.append(adjusted_sentence)
        entities.entities = adjusted_entities  # Replace with adjusted entities
        return entities
    
    
    
    
class NonStrictEntityAnalyzer(EntityErrorAnalyzer):
    """Analyzer for non-strict entity processing."""

    def extract_entities(self, y_data):
        """Extract entities in non-strict mode."""
        return [
            [(sen_id,) + entity for entity in get_entities(sen)]
            for sen_id, sen in enumerate(y_data)
        ]

    def prepare_entities(self):
        """Prepare true and predicted entities for non-strict mode."""
        self.true_entities = self.flatten_entities(self.extract_entities(self.y_true))
        self.pred_entities = self.flatten_entities(self.extract_entities(self.y_pred))

    def print_sentence(self, sen_id):
        """Print entities for a specific sentence ID."""
        true_entities = self.extract_entities(self.y_true)
        pred_entities = self.extract_entities(self.y_pred)
        print(f"True: {true_entities[sen_id]}")
        print(f"Pred: {pred_entities[sen_id]}")
        error = set(pred_entities[sen_id]) - set(true_entities[sen_id])
        print(f"Error in Pred: {error}")
        core_data = self.df[self.df['Labels'] !=-100]
        sentence_data = core_data[core_data['Sentence Ids']  == sen_id].copy()
        print(sentence_data[['Words', 'Sentence Ids', 'True Labels', 'Pred Labels', 'Strict True Entities', 'Strict Pred Entities', 'True Entities', 'Pred Entities']].head(60).to_string())
        
    @staticmethod
    def flatten_entities(entities):
        """Flatten non-strict entities into tuples."""
        return [e for sen in entities for e in sen]

class ErrorAnalysisManager:
    """Manages all error analysis workflows and stores results."""

    def __init__(self, df):
        """
        Initialize the manager with the dataset.

        Args:
            df (pd.DataFrame): The dataset containing y_true and y_pred.
        """
        self.df = df
        self.strict_analyzer = StrictEntityAnalyzer(df)
        self.non_strict_analyzer = NonStrictEntityAnalyzer(df)
        self.results = {
            "IOB2": {"false_negatives": None, "false_positives": None, "errors": None},
            "IOB": {"false_negatives": None, "false_positives": None, "errors": None},
        }

    def run_workflows(self):
        """Run all error analysis workflows."""
        self.results["IOB2"]["false_negatives"] = self.strict_analyzer.analyze_component("false_negatives")
        self.results["IOB2"]["false_positives"] = self.strict_analyzer.analyze_component("false_positives")
        self.results["IOB2"]["errors"] = self.strict_analyzer.analyze_errors()

        self.results["IOB"]["false_negatives"] = self.non_strict_analyzer.analyze_component("false_negatives")
        self.results["IOB"]["false_positives"] = self.non_strict_analyzer.analyze_component("false_positives")
        self.results["IOB"]["errors"] = self.non_strict_analyzer.analyze_errors()

    def get_results(self):
        """Get the results of all workflows."""
        return self.results

class SchemeComparator:
    """Facilitator for comparing annotation schemes."""

    def __init__(self, results):
        """
        Initialize the comparator with results from error analysis.

        Args:
            results (dict): Results from the manager's workflows, structured by scheme.
        """
        self.results = results

    def compare_component(self, component, entity_type):
        """
        Compare all error types for a specific entity across schemes.

        Args:
            entity_type (str): The entity type to compare (e.g., "MISC").

        Returns:
            dict: A dictionary with set operation results for all error types.
        """
        schemes = list(self.results.keys())
        if len(schemes) != 2:
            raise ValueError("Comparator requires exactly two schemes for comparison.")

        scheme_1, scheme_2 = schemes
        component_1 = self.results[scheme_1][component]
        component_2 = self.results[scheme_2][component]

        results = {}
        entity_1 = component_1.get(entity_type, {})
        entity_2 = component_2.get(entity_type, {})

        # Compare all error types under the given entity
        all_error_types = set(entity_1.keys()).union(set(entity_2.keys()))
        for error_type in all_error_types:
            set_1 = set(entity_1.get(error_type, []))
            set_2 = set(entity_2.get(error_type, []))

            results[error_type] = {
                "overlap": set_1 & set_2,
                f"{scheme_1} Only": set_1 - set_2,
                f"{scheme_2} Only": set_2 - set_1,
            }

        return results

    def compare_errors(self, component, error_type):
        """
        Compare errors across all entities and error types for both schemes.

        Returns:
            dict: A dictionary with set operation results for all error types.
        """
        schemes = list(self.results.keys())
        if len(schemes) != 2:
            raise ValueError("Comparator requires exactly two schemes for comparison.")

        schemes_map = {'scheme_1': 'IOB', 'scheme_2': 'IOB2'}
        errors_1 = self.results[schemes_map['scheme_1']]["errors"][component]
        errors_2 = self.results[schemes_map['scheme_2']]["errors"][component]

       
       
        comparison_result = ComparisonResult.from_lists(errors_1, errors_2, error_type, schemes_map)

        return comparison_result.to_dict()


@dataclass
class ComparisonResult:
    """Dataclass to store comparison results."""
    scheme_1_name: str
    scheme_2_name: str
    set_1_errors: Set[int] = field(default=set)
    set_2_errors: Set[int] = field(default=set)
    overlap: Set[int] = field(default_factory=set)
    scheme_1_only: Set[int] = field(default_factory=set)
    scheme_2_only: Set[int] = field(default_factory=set)

    @staticmethod
    def from_lists(errors_1: Dict, errors_2: Dict, error_type: str, schemes_map: Dict) -> "ComparisonResult":
        """
        Create a ComparisonResult from two lists.

        Args:
            lst_1: List of values from scheme 1.
            lst_2: List of values from scheme 2.

        Returns:
            ComparisonResult: Dataclass containing the comparison and statistics.
        """
        set_1 = set(errors_1.get(error_type, []))
        
        set_2 = set(errors_2.get(error_type, []))
        
        sentence_lst_1 = [error[0] for error in errors_1.get(error_type, [])]
        sentence_lst_2 = [error[0] for error in errors_2.get(error_type, [])]
        sentence_set_1 = set(sentence_lst_1)
        sentence_set_2 = set(sentence_lst_2)
        
        overlap = sentence_set_1 & sentence_set_2
        scheme_1_only = sentence_set_1 - sentence_set_2
        scheme_2_only = sentence_set_2 - sentence_set_1

        return ComparisonResult(
            scheme_1_name=schemes_map['scheme_1'],
            scheme_2_name=schemes_map['scheme_2'],
            set_1_errors= set_1,
            set_2_errors= set_2,
            overlap=overlap,
            scheme_1_only=scheme_1_only,
            scheme_2_only=scheme_2_only,
        )
        
    def to_dict(self) -> Dict[str, Dict[str, Set[int]]]:
        """R"Overlap": self.overlap, comparison results as a dictionary."""
        return {
            f"{self.scheme_1_name} Errors": self.set_1_errors,
            f"{self.scheme_2_name} Errors": self.set_2_errors,
            "Overlap": self.overlap,
            f"{self.scheme_1_name} Only Errors": self.scheme_1_only,
            f"{self.scheme_2_name} Only Errors": self.scheme_2_only,
        }
