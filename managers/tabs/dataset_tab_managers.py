import logging
import json
import pandas as pd
from config.enums import (CorrelationCoefficients, CustomAnalysisType,
                          ResultsType)
from layouts.managers.layout_managers import CustomDataTable
from managers.plotting.dataset_plotting_managers import (
    CorrelationAnalysis, DistributionAnalysis, ErrorRateAnalysis,
    SentenceLengthPlot, TagAmbiguityAnalysis, TokenLengthPlot,
    VariabilityAnalysis)
from managers.tabs.tab_managers import BaseTabManager


class DatasetTabManager(BaseTabManager):
    def __init__(self, variants_data):
        super().__init__(variants_data)

    def load_corpora(self, corpus_name='ANERCorp_CamelLab'):
        with open('/Users/ay227/Library/CloudStorage/GoogleDrive-ahmed.younes.sam@gmail.com/My Drive/Final Year Experiments/Thesis-Experiments/Experiments/ExperimentData/corpora.json', 'r') as file:
            corpora = json.load(file)
        df = pd.DataFrame([{'Word':w, 'Tag':t} for data in corpora[corpus_name]['splits']['test'] for w, t in zip(data['words'], data['tags'])])
        return df
    
    def generate_statistics(self, data, columns):
        """Generate statistical summaries for given columns."""
        if not columns:
            return None

        valid_columns = [col for col in columns if col in data.columns]
        if not valid_columns:
            return None

        statistics_df = (
            data[valid_columns]
            .describe()
            .reset_index()
            .rename(columns={"index": "Statistics"})
        )
        return statistics_df

    def generate_summary_statistics(self, variant, statistical_columns):
        """Generates summary statistics and handles the output for the summary table."""
        tab_data = self.get_tab_data(variant)
        if not tab_data:
            return None  # Indicate that no data was found

        filtered_data = self.filter_ignored(tab_data.analysis_data)

        data = self.generate_statistics(filtered_data, statistical_columns)
        if data is None or data.empty:
            return None  # Indicate that the data is empty

        return CustomDataTable(
            table_id="summary_data_table",
            data=data.to_dict("records"),
            columns=[{"name": col, "id": col} for col in data.columns],
        ).render()

    def generate_distribution_or_violin(
        self, variant, distribution_column, categorical_column
    ):
        tab_data = self.get_tab_data(variant)
        selected_df = self.filter_ignored(tab_data.analysis_data)

        if not distribution_column or not tab_data:
            return None

        # Delegate to PlottingAnalysis
        plotting_analysis = DistributionAnalysis()
        figure = plotting_analysis.generate_distribution_or_violin(
            selected_df, distribution_column, categorical_column
        )

        return figure

    def get_results_data(self, variant, results_type):
        """Fetch results data for a specific variant and results type."""
        tab_data = self.get_tab_data(variant)

        if not tab_data:
            logging.error("No data available for the selected variant.")
            return None

        try:
            results_type_enum = ResultsType(results_type)
        except ValueError:
            logging.error("Invalid results type selected.")
            return None

        if results_type_enum == ResultsType.TRAINING:
            data = tab_data.results
        elif results_type_enum == ResultsType.CLUSTERING:
            data = tab_data.kmeans_results
        elif results_type_enum == ResultsType.TOKEN:
            data = tab_data.token_report
        elif results_type_enum == ResultsType.ENTITY:
            data = tab_data.entity_non_strict_report
        elif results_type_enum == ResultsType.STRICT_ENTITY:
            data = tab_data.entity_strict_report
        else:
            logging.error("Unknown results type.")
            return None

        if data is None or data.empty:
            logging.warning("No results data available for the selected criteria.")
            return None

        return CustomDataTable(
            table_id="results_data_table",
            data=data.to_dict("records"),
            columns=[{"name": col, "id": col} for col in data.columns],
        ).render()

    def calculate_correlation(
        self,
        variant,
        correlation_method,
        categorical_column,
        x_column="Inconsistency Count",
        y_column="Inconsistency Count",
    ):
        """
        Calculate and return correlation matrix and scatter plot for selected data.
        """
        tab_data = self.get_tab_data(variant)

        if not tab_data:
            logging.error("No data available for the selected variant.")
            return None, None

        selected_df = self.filter_ignored(tab_data.analysis_data)
        if selected_df.empty:
            logging.error("No relevant data available after filtering.")
            return None, None

        try:
            coefficient_type_enum = CorrelationCoefficients(correlation_method)
        except ValueError:
            logging.error("Invalid coefficient selected.")
            return None, None

        if coefficient_type_enum == CorrelationCoefficients.PEARSON:
            coefficient = "pearson"
        elif coefficient_type_enum == CorrelationCoefficients.SPEARMAN:
            coefficient = "spearman"
        else:
            logging.error("Unknown coefficient.")
            return None, None

        correlation_analysis = CorrelationAnalysis()

        return correlation_analysis.calculate_correlation(
            selected_df, coefficient, categorical_column, x_column, y_column
        )

    def create_token_variability_table(self, selected_df):
        """
        Calls the TokenDistributionAnalysis class to create a variability table.
        """
        variability_analysis = VariabilityAnalysis()
        token_variability = variability_analysis.create_token_variability_table(selected_df)
        return token_variability
    
    def create_word_variability_table(self, corpus_name):
        """
        Calls the TokenDistributionAnalysis class to create a variability table.
        """
        variability_analysis = VariabilityAnalysis()
        corpus = self.load_corpora(corpus_name)
        word_variability = variability_analysis.create_word_variability_table(corpus)        
        return word_variability

    def create_tag_ambiguity_table(self, selected_df):
        """
        Calls the TagAmbiguityAnalysis class to create a tag ambiguity table.
        """
        tag_ambiguity_analysis = TagAmbiguityAnalysis()
        return tag_ambiguity_analysis.create_tag_ambiguity_table(selected_df)

    def create_token_length_plot(self, selected_df):
        """
        Calls the TagAmbiguityAnalysis class to create a tag ambiguity table.
        """
        plotter = TokenLengthPlot()
        figure = plotter.generate_plot(selected_df)
        return figure

    def create_sentence_length_plot(self, selected_df):
        """
        Calls the TagAmbiguityAnalysis class to create a tag ambiguity table.
        """
        plotter = SentenceLengthPlot()
        figure = plotter.generate_plot(selected_df)
        return figure

    def create_weighted_error_rate_plot(self, selected_df):
        """
        Calls the DistributionAnalysis class to create a weighted error rate plot.
        """
        error_rate_analysis = ErrorRateAnalysis()
        figure = error_rate_analysis.generate_weighted_error_rate_plot(
            selected_df,
        )
        return figure

    def perform_custom_analysis(self, custom_distribution_type, variant, corpus_name):

        try:
            analysis_type_enum = CustomAnalysisType(custom_distribution_type)
        except ValueError:
            logging.error("Invalid distribution type selected.")
            return None

        tab_data = self.get_tab_data(variant)
        if not tab_data:
            logging.error("No tab data available.")
            return None

        selected_df = self.filter_ignored(tab_data.analysis_data)
        if selected_df.empty:
            logging.error("No relevant data available after filtering.")
            return None

        if analysis_type_enum == CustomAnalysisType.TOKEN:
            analysis = self.create_token_variability_table(selected_df)
        elif analysis_type_enum == CustomAnalysisType.WORD:
            analysis = self.create_word_variability_table(corpus_name)
        elif analysis_type_enum == CustomAnalysisType.TAG:
            analysis = self.create_tag_ambiguity_table(selected_df)
        elif analysis_type_enum == CustomAnalysisType.TOKEN_LENGTH:
            analysis = self.create_token_length_plot(selected_df)
        elif analysis_type_enum == CustomAnalysisType.SENTENCE_LENGTH:
            analysis = self.create_sentence_length_plot(selected_df)
        elif analysis_type_enum == CustomAnalysisType.TOKENIZATION_ERROR_RATE:
            analysis = self.create_weighted_error_rate_plot(selected_df)
        else:
            logging.error("Invalid distribution type.")
            analysis = None

        return analysis
