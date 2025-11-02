import logging
import json
import pandas as pd
from dash import html
from config.enums import (CorrelationCoefficients, CustomAnalysisType,
                          ResultsType)
from layouts.managers.layout_managers import CustomDataTable
from config.config_managers import ColorMap
from managers.plotting.cross_component_plotting_managers import (
    render_dataset_stats_table,
    plot_faceted_bar_chart,
    plot_entity_span_distribution,
    plot_entity_span_complexity_box,
    plot_colored_bar_chart,
    render_oov_row_table,
    plot_entity_tag_oov_bar,
    plot_overlap_heatmaps,
    plot_token_behaviour_bar,
    plot_confidence_heatmaps_px,
    render_eval_overall_table,
    plot_metric_bar,
    plot_span_confusion_heatmap, 
    plot_entity_errors_heatmap,
    plot_token_confusion_heatmap,
    plot_support_corr_heatmaps, 
    plot_support_vs_metric_scatter,
    plot_spearman_rankdiff_bars,
    ttr,
    )
from managers.tabs.tab_managers import BaseTabManager
from pathlib import Path
from managers.tabs.cross_component_helpers import (
    # Data Component Helpers 
    DatasetStatsHelper, EntityTagDistribution, WordTypeFrequencyDistribution, 
    EntitySpanDistribution, EntitySpanComplexity,
    TokenisedDatasetStatsHelper, EntityTagTokenTypeDistribution, TokenTypeFrequencyDistribution, 
    DatasetOOVRate, EntityTagOOVRate,
    DatasetTokenOOVRate, EntityTagTokenOOVRate, 
    WordTypeOverlap, TokenTypeOverlap, 
    TokenizationRateHelper, AmbiguityHelper, ConsistencyHelper, 
    _DEFAULT_TAGS, _DEFAULT_ENTITY_SPANS,

    # Model Component Helpers
    LossHelper, PredictionUncertaintyHelper, 
    PerClassConfidenceHelper, TokenConfidenceHelper, ConfidenceConfusionHelper, 
    SilhouetteHelper,

    # Evaluation Helpers
    TokenVsEntityOverallHelper, EntitySchemesOverallHelper,
    EntitySpanF1Helper, EntitySpanPRHelper, EntitySpanSupportHelper,
    EntitySpanPredictionOutcomeBreakdownHelper, EntitySpanConfusionHelper,
    SpanErrorTypesHelper, SpanErrorTypesHeatmapHelper, SpanEntityErrorsHeatmapHelper,
    TokenF1Helper, TokenPrecisionRecallHelper, TokenSupportHelper, TokenPredictionOutcomesHelper,
    TokenMisclassHeatmapHelper, 
    TokenSupportCorrelationHelper, TokenSupportScatterHelper, TokenSpearmanHelper

    )



class DataTabManager(BaseTabManager):
    def __init__(self, variants_data, config_manager):
        super().__init__(variants_data)
        self.corpora_path = Path(config_manager.corpora_dir)
        self.corpora = self.load_corpora()
        self.dataset_helper = DatasetStatsHelper(self.corpora)
        self.entity_tag_helper = EntityTagDistribution(self.corpora)
        self.entity_span_helper = EntitySpanDistribution(self.corpora)
        self.entity_span_complexity_helper = EntitySpanComplexity(self.corpora)  
        self.entity_tag_word_type_frequency_helper = WordTypeFrequencyDistribution(self.corpora)
        self.dataset_oov_rate_helper = DatasetOOVRate(self.corpora)
        self.entity_tag_oov_rate_helper = EntityTagOOVRate(self.corpora)
        self.tokenised_stats_helper = TokenisedDatasetStatsHelper(variants_data)
        self.entity_tag_token_type_helper = EntityTagTokenTypeDistribution(variants_data)
        self.token_type_freq_helper = TokenTypeFrequencyDistribution(variants_data)
        self.dataset_token_oov_rate_helper = DatasetTokenOOVRate(variants_data)
        self.entity_tag_token_oov_rate_helper = EntityTagTokenOOVRate(variants_data)
        self.word_overlap_helper = WordTypeOverlap(self.corpora)
        self.token_overlap_helper = TokenTypeOverlap(variants_data)
        self.tokenisation_rate_helper = TokenizationRateHelper(variants_data)
        self.ambiguity_helper = AmbiguityHelper(variants_data)
        self.consistency_helper = ConsistencyHelper(variants_data)


    def load_corpora(self):
        with open(self.corpora_path /'corpora.json', 'r') as file:
            corpora = json.load(file)
        return corpora
    
    def get_corpus(self, corpus_name='ANERCorp_CamelLab'):
        df = pd.DataFrame([{'Word':w, 'Tag':t} for data in self.corpora[corpus_name]['splits']['test'] for w, t in zip(data['words'], data['tags'])])
        return df
    
    # TRANSFORMATION ONLY: returns a pandas DataFrame (raw values)
    def generate_dataset_stats(self, selected_variant: str) -> html.Div:
        df = self.dataset_helper.generate_df(selected_variant)
        return render_dataset_stats_table(df)
    
    def generate_entity_tag_distribution(self, selected_variant: str) -> html.Div:
        df = self.entity_tag_helper.generate_df(selected_variant)
        return plot_faceted_bar_chart(
            df=df,
            metric="Words Proportion",
            text="Tag Words",
            title="Entity Tag Words Distribution Across Training and Testing Splits",
            tag_order=_DEFAULT_TAGS
        )
    
    def generate_entity_span_distribution(self, selected_variant: str) -> html.Div:
        df = self.entity_span_helper.generate_df(selected_variant)
        return plot_entity_span_distribution(
            df=df,
            metric="Span Proportion",
            text="Span Count",
            title="Entity Span Distribution Across Schemes, Splits and Datasets",
            tag_order=_DEFAULT_ENTITY_SPANS,
        )
    

    def generate_entity_span_complexity_distribution(self, selected_variant: str) -> html.Div:
        df = self.entity_span_complexity_helper.generate_df(selected_variant)
        return plot_entity_span_complexity_box(
            df,
            title="Entity Span Length Distribution (i.e., Complexity as Number of Words) Across Annotation Schemes, Splits, and Datasets",
            tag_order=_DEFAULT_ENTITY_SPANS,
        )
    
    def generate_entity_tag_type_distribution(self, selected_variant: str) -> html.Div:
        df = self.entity_tag_helper.generate_df(selected_variant)
        return plot_faceted_bar_chart(
            df=df,
            metric="Tag Type Proportion",
            text="Tag Types",
            title="Entity Tag Word Types Distribution Across Training and Testing Splits",
            tag_order=_DEFAULT_TAGS
        )
    
    def generate_type_to_word_ratio(self, selected_variant: str) -> html.Div:
        df = self.entity_tag_helper.generate_df(selected_variant)
        return plot_faceted_bar_chart(
            df=df,
            metric='TWR',
            text='TWR',
            title='Type-to-Word Ratio (TWR) Across Entity Tags in Arabic and Engilsh (Train and Test Splits)',
            tag_order=_DEFAULT_TAGS,
        )
    
    def generate_word_type_frequency_distribution(self, selected_variant: str) -> html.Div:
        df = self.entity_tag_word_type_frequency_helper.generate_df(selected_variant)
        return plot_colored_bar_chart(
            df=df,
            metric="Standard Deviation",
            title="Std. Dev. of Word-Type Frequencies Across Entity Tags (Train/Test)",
            tag_order=_DEFAULT_TAGS,
        )

    def generate_dataset_oov_rate(self, selected_variant: str) -> html.Div:
        df = self.dataset_oov_rate_helper.generate_df(selected_variant)
        return render_oov_row_table(df)
    
    def generate_entity_tag_oov_rate(self, selected_variant: str) -> html.Div:
        df = self.entity_tag_oov_rate_helper.generate_df(selected_variant)
        return plot_entity_tag_oov_bar(
            df,
            title="Entity Tag Word Level OOV Rate Across Arabic and English",
            tag_order=_DEFAULT_TAGS,
            show_values=True,
        )
    
    def generate_tokenised_dataset_stats(self, selected_variant: str) -> html.Div:
        """
        Build tokens-only stats (Total/Unique/NE/TWR/TEWR) for the selected variant
        using the TokenisedDatasetStatstHelper and render as a table.
        """
        df = self.tokenised_stats_helper.generate_df(selected_variant)
        return render_dataset_stats_table(df)
    
    def generate_entity_tag_token_type_distribution(self, selected_variant: str) -> html.Div:
        df = self.entity_tag_token_type_helper.generate_df(selected_variant)
        return plot_faceted_bar_chart(
            df=df,
            metric="Type Proportion",        # or "TTR" depending on the figure
            text="Tag Types",
            title="Entity Tag Token Types Distribution Across Training and Testing Splits",
            tag_order=_DEFAULT_TAGS,
        )

    def generate_type_to_token_ratio(self, selected_variant: str) -> html.Div:
        df = self.entity_tag_token_type_helper.generate_df(selected_variant)        
        # reuse the same plot function, switch metric
        return ttr(
            df=df,
            height=700, width=1400,
            tag_order=_DEFAULT_TAGS,
        )

    def generate_token_type_frequency_distribution(self, selected_variant: str) -> html.Div:
        df = self.token_type_freq_helper.generate_df(selected_variant)
        return plot_colored_bar_chart(
            df=df,
            metric="Standard Deviation",
            title="Std. Dev. of Token-Type Frequencies Across Entity Tags (Train/Test)",
            tag_order=_DEFAULT_TAGS,
        )
    
    # def generate_dataset_token_oov_rate(self, selected_variant: str) -> html.Div:
    #     df = self.dataset_token_oov_rate_helper.generate_df(selected_variant)  # rows=metrics, cols=datasets
    #     return render_oov_row_table(df)  

    def generate_entity_tag_token_oov_rate(self, selected_variant: str) -> html.Div:
        df = self.entity_tag_token_oov_rate_helper.generate_df(selected_variant)
        # proportion on y; counts as text labels
        return plot_entity_tag_oov_bar(
            df,
            title="Entity Tag Token Level OOV Rate Across Arabic and English",
            tag_order=_DEFAULT_TAGS,
            percent_axis=False,   # keep proportions as decimals; set True if you want % axis
            text_digits=3,        # or 3, your call
        )
    

    def generate_word_type_overlap_train(self, selected_variant: str) -> html.Div:
        df = self.word_overlap_helper.generate_df(selected_variant)
        return plot_overlap_heatmaps(
            df,
            title="Word Type Overlap (Train Split)",
            panel_by="Language",
            filter_equals={"Split": "Train"},
            tag_order=_DEFAULT_TAGS + ["O"],
        )

    def generate_word_type_overlap_test(self, selected_variant: str) -> html.Div:
        df = self.word_overlap_helper.generate_df(selected_variant)
        return plot_overlap_heatmaps(
            df,
            title="Word Type Overlap (Test Split)",
            panel_by="Language",
            filter_equals={"Split": "Test"},
            tag_order=_DEFAULT_TAGS + ["O"],
        )

    def generate_token_type_overlap_train(self, selected_variant: str) -> html.Div:
        df = self.token_overlap_helper.generate_df(selected_variant)
        return plot_overlap_heatmaps(
            df,
            title="Token Type Overlap (Train Split)",
            panel_by="Language",
            filter_equals={"Split": "Train"},
            tag_order=_DEFAULT_TAGS + ["O"],
        )

    def generate_token_type_overlap_test(self, selected_variant: str) -> html.Div:
        df = self.token_overlap_helper.generate_df(selected_variant)
        return plot_overlap_heatmaps(
            df,
            title="Token Type Overlap (Test Split)",
            panel_by="Language",
            filter_equals={"Split": "Test"},
            tag_order=_DEFAULT_TAGS + ["O"],
        )
    
    def generate_tokenization_rate(self, selected_variant: str) -> html.Div:
        df = self.tokenisation_rate_helper.generate_df(selected_variant)
        return plot_token_behaviour_bar(
            df,
            title="Tokenisation Rate per Entity Tag",
            facet_by_level=False,
            facet_by_split=False,
            tag_order=_DEFAULT_TAGS + ["O"],
            y_axis_label="Average Score",
        )

    def generate_ambiguity(self, selected_variant: str) -> html.Div:
        df = self.ambiguity_helper.generate_df(selected_variant)
        return plot_token_behaviour_bar(
            df,
            title="Ambiguity by Entity Tag (Token vs Word; Mean ± SD)",
            facet_by_level=True,            # facet rows by Level
            facet_by_split=False,           # no column facets
            tag_order=_DEFAULT_TAGS + ["O"],
            level_order=["Token Level", "Word Level"],
            y_axis_label="Average Score",
            color_col="Level", 
            facet_row_col="Language",   
        )

    def generate_consistency_metrics(self, selected_variant: str) -> html.Div:
        df = self.consistency_helper.generate_df(selected_variant)
        return plot_token_behaviour_bar(
            df,
            title="Consistency vs Inconsistency by Entity Tag (Mean ± SD)",
            facet_by_level=True,
            facet_by_split=False,
            tag_order=_DEFAULT_TAGS + ["O"],
            level_order=["Consistency Ratio", "Inconsistency Ratio"],
            y_axis_label="Average Score",
            color_col="Level", 
            facet_row_col="Language",   
        )



class ModelTabManager(BaseTabManager):
    def __init__(self, variants_data, config_manager):
        super().__init__(variants_data)
        self.loss_helper = LossHelper(variants_data)
        self.silhouette_helper = SilhouetteHelper(variants_data)
        self.prediction_uncertainty_helper = PredictionUncertaintyHelper(variants_data)
        self.per_class_conf_helper = PerClassConfidenceHelper(variants_data)
        self.token_conf_helper = TokenConfidenceHelper(variants_data)
        self.conf_matrix_helper = ConfidenceConfusionHelper(variants_data)   


    def generate_loss(self, selected_variant: str) -> html.Div:
        df = self.loss_helper.generate_df(selected_variant)
        return plot_token_behaviour_bar(
            df,
            title="Loss Values Per Entity Tag (Mean ± SD)",
            facet_by_level=False,
            facet_by_split=False,
            tag_order=_DEFAULT_TAGS + ["O"],
            y_axis_label="Average Loss",
        )
    
    def generate_silhouette(self, selected_variant: str) -> html.Div:
        df = self.silhouette_helper.generate_df(selected_variant)
        return plot_token_behaviour_bar(
            df,
            title="Silhouette Score by Tag (True vs Pred; Mean ± SD)",
            facet_by_level=True,                 # two bars per tag: True vs Pred
            facet_by_split=False,                # usually just Test
            tag_order=_DEFAULT_TAGS + ["O"],
            level_order=["True Silhouette", "Pred Silhouette"],
            color_col="Level", 
            y_axis_label="Silhouette Score",     # range is typically [-1, 1]
            facet_row_col="Language",      # facet rows by Language
        )
    
    def generate_prediction_uncertainty(self, selected_variant: str) -> html.Div:
        df = self.prediction_uncertainty_helper.generate_df(selected_variant)
        return plot_token_behaviour_bar(
            df,
            title="Entity Tag Prediction Uncertainty (Correct vs Error; Mean ± SD)",
            tag_order=_DEFAULT_TAGS + ["O"],
            level_order=["Correct", "Error"],     # color legend order
            color_col="Type",                     # colors reflect Correct vs Error
            y_axis_label="Prediction Uncertainty",
            # facet rows by Language (clean small multiples per language)
            facet_row_col="Language",
            # no split facets (we're using Test only via helper)
            facet_by_level=False,
            facet_by_split=False,
            decimals=3,
        )
    
    def generate_pre_class_confidence(self, selected_variant: str) -> html.Div:
        df = self.per_class_conf_helper.generate_df(selected_variant, include_all=False)
        return plot_token_behaviour_bar(
            df,
            title="Per-Class Confidence Scores Across Entity Tags and Correct and Error Predictions (Mean ± SD)",
            tag_order=_DEFAULT_TAGS + ["O"],
            level_order=["Correct", "Error"],   # optional: include "All" if you passed include_all=True
            color_col="Type",                   # colors = Correct vs Error
            y_axis_label="Average Confidence",
            facet_row_col="Language",
            facet_by_level=False,
            facet_by_split=False,
            decimals=3,
        )
    
    def generate_token_confidence(self, selected_variant: str) -> html.Div:
        df = self.token_conf_helper.generate_df(selected_variant, include_all=False)
        return plot_token_behaviour_bar(
            df,
            title="Token Confidence Per Entity Tag Across Correct and Error (Mean ± SD)",
            tag_order=_DEFAULT_TAGS + ["O"],
            level_order=["Correct", "Error"],  # Agreements split
            color_col="Type",
            y_axis_label="Average Confidence",
            facet_row_col="Language",
            facet_by_level=False,
            facet_by_split=False,
            decimals=3,
        )

    def generate_confidence_confusion(self, selected_variant: str) -> html.Div:
        df = self.conf_matrix_helper.generate_df(
            selected_variant,
            agg="sum",        # or "sum"
            round_to=2,
            only_test=True
        )
        return plot_confidence_heatmaps_px(
            df,
            title="Total Token Confidence Confusion Heatmap",
            panel_by="Language",
            value_col="Overlap Count",
            tag_order=_DEFAULT_TAGS + ["O"],
            colorscale="RdBu_r",
            value_precision=2,
            shared_scale=True,
        )
    

HEIGHT = 700
WIDTH = 1400

class EvaluationTabManager(BaseTabManager):
    def __init__(self, variants_data, config_manager):
        super().__init__(variants_data)
        self.color_map = ColorMap().color_map
        self.token_vs_entity_helper = TokenVsEntityOverallHelper(variants_data)
        self.entity_schemes_helper = EntitySchemesOverallHelper(variants_data)
        self.entity_f1_helper = EntitySpanF1Helper(variants_data)
        self.span_pr_rec_helper = EntitySpanPRHelper(variants_data)
        self.span_support_helper = EntitySpanSupportHelper(variants_data)
        self.span_prediction_outcome_helper = EntitySpanPredictionOutcomeBreakdownHelper(variants_data)
        self.span_confusion_helper = EntitySpanConfusionHelper(variants_data)
        self.span_error_types_helper = SpanErrorTypesHelper(variants_data)
        self.span_error_types_heatmap_helper = SpanErrorTypesHeatmapHelper(variants_data)
        self.span_entity_errors_helper = SpanEntityErrorsHeatmapHelper(variants_data)
        self.token_f1_helper   = TokenF1Helper(variants_data)
        self.token_pr_rec_helper   = TokenPrecisionRecallHelper(variants_data)
        self.token_support_helper  = TokenSupportHelper(variants_data)
        self.token_prediction_outcome_helper  = TokenPredictionOutcomesHelper(variants_data)
        self.token_misclass_helper  = TokenMisclassHeatmapHelper(variants_data)
        self.token_support_corr_helper  = TokenSupportCorrelationHelper(variants_data)
        self.token_support_scatter_helper = TokenSupportScatterHelper(variants_data)
        self.token_spearman_helper = TokenSpearmanHelper(variants_data)
        
        



    def generate_overall_token_vs_entity(self, selected_variant: str):
        df = self.token_vs_entity_helper.generate_df(selected_variant, round_to=4)
        return render_eval_overall_table(df, digits=4)
    
    def generate_overall_entity_schemes(self, selected_variant: str):
        df = self.entity_schemes_helper.generate_df(selected_variant, round_to=4)
        return render_eval_overall_table(df, digits=4)
    
    def generate_span_f1(self, selected_variant: str):
        df = self.entity_f1_helper.generate_df(selected_variant, round_to=3)
        return plot_metric_bar(
            df,
            metric_col="F1-score",
            title="F1 by Entity Span across Languages and Annotation Schemes (IOB1 vs IOB2)",
            color="Scheme",        # compare IOB1 vs IOB2
            facet_col="Language",  # one row per language/model
            x_axis_title='Entity Span',
            height=HEIGHT,
            width=WIDTH,
            y_as_percent=False
        )
    
    def generate_span_precision_recall(self, selected_variant: str) -> html.Div:
        df = self.span_pr_rec_helper.generate_df(selected_variant, round_to=3)
        return plot_metric_bar(
            df=df,
            metric_col="Value",
            title="Precision & Recall by Entity Span across Languages and Annotation Schemes (IOB1 vs IOB2)",
            color="Metric",          # Precision vs Recall are colored
            facet_row="Scheme",      # IOB1 / IOB2 in rows
            facet_col="Language",    # Arabic / English in columns
            x_axis_title='Entity Span',
            y_axis_title='Score',
            height=HEIGHT,
            width=WIDTH,
            text_round=3
        )
    
    def generate_span_support(self, selected_variant: str):
        df = self.span_support_helper.generate_df(selected_variant, round_to=3)
        return plot_metric_bar(
            df=df,
            metric_col="Support",
            title="Support Counts by Entity Span across Languages and Annotation Schemes (IOB1 vs IOB2)",
            color="Scheme",         # color by scheme
            facet_col="Language",   # rows = Arabic / English
            height=HEIGHT,
            width=WIDTH,
            text_round=0            # integer support counts
        )
    
    def generate_span_prediction_outcomes(self, selected_variant: str):
        df = self.span_prediction_outcome_helper.generate_df(selected_variant)
        return plot_metric_bar(
            df=df,
            metric_col="Scale",   # proportions on Y
            title="Entity Span prediction outcomes (TP, FP, FN) by language and annotation scheme (IOB1 vs IOB2)",
            color="Metric",       # TP vs FP vs FN
            facet_row="Scheme",   # IOB1 / IOB2 rows
            facet_col="Language", # Arabic / English columns (or use Model if you prefer)
            text_col="Count",     # show raw counts on bars
            text_round=0,          # counts are integers
            height=HEIGHT,
            width=WIDTH,
            tag_order=_DEFAULT_ENTITY_SPANS,
        )
    
    def generate_span_prediction_errors(self, selected_variant: str) -> html.Div:
        df = self.span_confusion_helper.generate_df(selected_variant)
        return plot_span_confusion_heatmap(
            df=df,
            value_col="Count",                 # or "Share"
            text_col="Count",                  # show integers in cells
            row_by="Scheme",
            col_by="Language",
            tag_order=_DEFAULT_ENTITY_SPANS,
            metric_order=["FN", "FP"],
            colorscale="RdBu_r",
            title="Entity Span false positives and false negatives by Annotation Scheme and Language",
            height=HEIGHT,
            width=WIDTH,
        )
    
    def generate_span_error_types(self, selected_variant: str) -> html.Div:
        df = self.span_error_types_helper.generate_df(selected_variant)
        color_map = {
            "False Positives": "#E74C3C",  # Red
            "False Negatives": "#00CC96",  # Teal
        }
        return plot_metric_bar(
            df=df,                   # from your ErrorType helper
            metric_col="Percentage",
            title="Distribution of Error Types within FP/FN across Langauges and Annotation Schemes (IOB1 vs IOB2)",
            x_col="Error Type",              # <— key change
            x_axis_title='Error Type',
            color="Component",               # False Positives / False Negatives
            facet_row="Scheme",
            facet_col="Language",
            text_col="Count",          # show raw counts as labels
            color_map=color_map,   # << now works
            height=HEIGHT, width=WIDTH, text_round=0
        )
    
    def generate_span_error_types_fp(self, selected_variant: str) -> html.Div:
        df = self.span_error_types_heatmap_helper.generate_df(selected_variant, "false_positives")
        return plot_span_confusion_heatmap(
            df=df,
            value_col="Count",
            text_col="Count",
            row_by="Scheme",
            col_by="Language",
            tag_order=_DEFAULT_ENTITY_SPANS,
            metric_order=["Entity", "Boundary", "Entity and Boundary", "O (Inclusion)"],
            title="False Positive Error Types by Entity Span (rows=Scheme, cols=Language)",
            colorscale="RdBu_r",
            height=HEIGHT,
            width=WIDTH,
        )

    def generate_span_error_types_fn(self, selected_variant: str) -> html.Div:
        df = self.span_error_types_heatmap_helper.generate_df(selected_variant, "false_negatives")
        return plot_span_confusion_heatmap(
            df=df,
            value_col="Count",
            text_col="Count",
            row_by="Scheme",
            col_by="Language",
            tag_order=_DEFAULT_ENTITY_SPANS,
            metric_order=["Entity", "Boundary", "Entity and Boundary", "O (Exclusion)"],
            title="False Negative Error Types by Entity Span (rows=Scheme, cols=Language)",
            colorscale="RdBu_r",
            height=HEIGHT,
            width=WIDTH,
        )


    # def generate_span_error_types_fp(self, selected_variant: str) -> html.Div:
    #     df = self.span_error_types_heatmap_helper.generate_df(selected_variant, "false_positives")
    #     return plot_span_confusion_heatmap(
    #         df=df,
    #         value_col="Count",
    #         text_col="Count",
    #         row_by="Scheme",
    #         col_by="Language",
    #         tag_order=_DEFAULT_ENTITY_SPANS,
    #         metric_order=["Entity","Boundary","Entity and Boundary","O Errors"],
    #         title="False Positives Error Type Heatmap: by Entity Span, Categorized by Language and Annotation Scheme",
    #         colorscale="RdBu_r",
    #         height=HEIGHT,
    #         width=WIDTH,
    #     )
    
    # def generate_span_error_types_fn(self, selected_variant: str) -> html.Div:
    #     df = self.span_error_types_heatmap_helper.generate_df(selected_variant, "false_negatives")
    #     return plot_span_confusion_heatmap(
    #         df=df,
    #         value_col="Count",
    #         text_col="Count",
    #         row_by="Scheme",
    #         col_by="Language",
    #         tag_order=_DEFAULT_ENTITY_SPANS,
    #         metric_order=["Entity","Boundary","Entity and Boundary","O Errors"],
    #         title="False Negatives Error Type Heatmap: by Entity Span, Categorized by Language and Annotation Scheme",
    #         colorscale="RdBu_r",
    #         height=HEIGHT,
    #         width=WIDTH,
    #     )
    
    def generate_span_entity_errors_fp(self, selected_variant: str) -> html.Div:
        df = self.span_entity_errors_helper.generate_df(selected_variant, component="false_positives")
        return plot_entity_errors_heatmap(
            df,
            title="False Positives Entity Errors by Entity Span (rows=Scheme, cols=Language)",
            row_by="Scheme",
            col_by="Language",
            tag_order=["LOC", "PER", "ORG", "MISC"],
            height=HEIGHT,
            width=WIDTH,
        )

    def generate_span_entity_errors_fn(self, selected_variant: str) -> html.Div:
        df = self.span_entity_errors_helper.generate_df(selected_variant, component="false_negatives")
        return plot_entity_errors_heatmap(
            df,
            title="False Negatives Entity Errors by Entity Span (rows=Scheme, cols=Language)",
            row_by="Scheme",
            col_by="Language",
            tag_order=["LOC", "PER", "ORG", "MISC"],
            height=HEIGHT,
            width=WIDTH,
        )

    def generate_token_f1(self, variant: str) -> html.Div:
        df = self.token_f1_helper.generate_df(variant, round_to=3)
        return plot_metric_bar(
            df=df, metric_col="F1-score", 
            title="Token-level F1-Score by Entity Tag Across Languages",
            x_axis_title="Entity Tag",
            facet_row="Language", tag_order=_DEFAULT_TAGS, y_as_percent=False,   # 👈 force decimals
        )

    # Token Precision & Recall
    def generate_token_pr_rc(self, variant: str) -> html.Div:
        df = self.token_pr_rec_helper.generate_df(variant, round_to=3)
        return plot_metric_bar(
            df=df, metric_col="Score", 
            title="Token-level Precision & Recall by Entity Tag Across Languages",
            x_axis_title="Entity Tag",
            color="Metric", facet_row="Language", tag_order=_DEFAULT_TAGS, y_as_percent=False,   # 👈 force decimals
        )

    # Token Support
    def generate_token_support(self, variant: str) -> html.Div:
        df = self.token_support_helper.generate_df(variant)
        return plot_metric_bar(
            df=df, metric_col="Support", 
            title="Token-level Support Counts by Entity Tag Across Languages",
            x_axis_title="Entity Tag",
            color=None, facet_row="Language", text_round=0,
            tag_order=_DEFAULT_TAGS, 
        )

    def generate_token_prediction_outcomes(self, variant: str) -> html.Div:
        df = self.token_prediction_outcome_helper.generate_df(variant, round_to=3)
        return plot_metric_bar(
            df=df,
            metric_col="Scaled Count",
            title="Token-level prediction outcomes (TP, FP, FN) by Language",
            x_col="Tag",
            color="Metric",
            facet_row="Language",
            text_col="Count",             # show raw count on bars
            tag_order=_DEFAULT_TAGS + ["O"],
            y_as_percent=False,
            height=HEIGHT, width=WIDTH,
            text_position="outside"
        )
    
    def generate_token_confusion_heatmap(self, variant: str) -> html.Div:
        df = self.token_misclass_helper.generate_df(variant)
        return plot_token_confusion_heatmap(
            df,
            panel_by="Language",
            tag_order=_DEFAULT_TAGS + ["O"],
            title="Token-level Misclassification Heatmap (True × Pred)"
        )
   
    def generate_token_support_correlations(self, variant: str) -> html.Div:
        df = self.token_support_corr_helper.generate_df(round_to=3)
        print(df)
        return plot_support_corr_heatmaps(df, height=HEIGHT, width=WIDTH)
    
    def generate_token_support_scatter(self, variant: str) -> html.Div:
        language = self.token_support_scatter_helper.ds_label(variant)

        points, means = self.token_support_scatter_helper.generate_for_language(variant, language, round_to=3)
        return plot_support_vs_metric_scatter(
            points_df=points, means_df=means,
            color_map=self.color_map, tag_order=_DEFAULT_TAGS,
            trendline="ols", height=HEIGHT, width=WIDTH,
        )
   
    def generate_token_spearman(self, variant: str) -> html.Div:
        language = self.token_support_scatter_helper.ds_label(variant)
        df = self.token_spearman_helper.generate_for_language(variant, language)
        return plot_spearman_rankdiff_bars(df, tag_order=_DEFAULT_TAGS, height=HEIGHT, width=WIDTH, language=language)



    # def generate_token_support_scatter_ar(self, variant: str) -> html.Div:
    #     points, means = self.token_support_scatter_helper.generate_df(variant, round_to=3)
    #     return plot_support_vs_metric_scatter(
    #         points_df=points,
    #         means_df=means,
    #         language="Arabic",
    #         color_map=self.color_map,           # you already have one
    #         tag_order=_DEFAULT_TAGS,            # if you keep a default tag order
    #         trendline="ols",
    #         height=HEIGHT, width=WIDTH,
    #     )


    # def generate_token_support_scatter_en(self, variant: str) -> html.Div:
    #     points, means = self.token_support_scatter_helper.generate_df(variant, round_to=3)
    #     return plot_support_vs_metric_scatter(
    #         points_df=points,
    #         means_df=means,
    #         language="English",
    #         color_map=self.color_map,
    #         tag_order=_DEFAULT_TAGS,
    #         trendline="ols",
    #         height=HEIGHT, width=WIDTH,
    #     )
    
    # def generate_token_spearman_ar(self, variant: str) -> html.Div:
    #     df = self.token_spearman_helper.generate_df(variant)
    #     return plot_spearman_rankdiff_bars(
    #         df, language="Arabic", tag_order=_DEFAULT_TAGS,
    #         height=HEIGHT, width=WIDTH
    #     )

    # def generate_token_spearman_en(self, variant: str) -> html.Div:
    #     df = self.token_spearman_helper.generate_df(variant)
    #     return plot_spearman_rankdiff_bars(
    #         df, language="English", tag_order=_DEFAULT_TAGS,
    #         height=HEIGHT, width=WIDTH
    #     )
