from dash import dcc, html

from layouts.managers.layout_managers import (CustomButton, DropdownContainer,
                                              LoadingContainer,
                                              SectionContainer, VariantSection,
                                              generate_dropdown_options)


class SummarySection:
    def __init__(self, config):
        self.statistical_dropdown = dcc.Dropdown(
            id="statistical_columns",
            multi=True,
            placeholder="Select column...",
            options=generate_dropdown_options(
                config.get("statistics_columns", ["Wrong Columns"])
            ),  # Assuming you have a function to generate options
        )
        self.generate_summary_button = CustomButton(
            "Generate Summary Statistics", "generate_summary"
        ).render()

    def render(self):

        return SectionContainer(
            "Summary Statistics",
            [
                self.statistical_dropdown,
                self.generate_summary_button,
            ],
        ).render()


class DistributionSection:
    def __init__(self, config):
        self.distribution_column_dropdown = dcc.Dropdown(
            id="distribution_column",
            multi=False,
            placeholder="Select Distribution column...",
            options=generate_dropdown_options(
                config.get("statistics_columns", ["Wrong Columns"])
            ),  # Assuming you have a function to generate options
            style={
                "width": "100%"
            },  # Assuming you want to use the full width for styling
        )
        self.categorical_column_dropdown = dcc.Dropdown(
            id="categorical_column",
            multi=False,
            placeholder="Select Categorical column...",
            options=generate_dropdown_options(
                config.get("categorical_columns", ["Wrong Columns"])
            ),  # Assuming you have a function to generate options
            style={"width": "100%"},
        )
        self.plot_button = CustomButton(
            "Plot Distribution", "plot_distribution"
        ).render()

    def render(self):
        return SectionContainer(
            "Distributions",
            [
                self.distribution_column_dropdown,
                self.categorical_column_dropdown,
                self.plot_button,  # Include the message
            ],
        ).render()


class ResultsSection:
    def __init__(self, config):
        self.results_dropdown = dcc.Dropdown(
            id="results_type",
            multi=False,
            placeholder="Select results...",
            options=generate_dropdown_options(
                config.get("results", [])
            ),  # Assuming you have a function to generate options
        )
        self.view_results_type_button = CustomButton(
            "View Results", "view_results_type"
        ).render()

    def render(self):

        return SectionContainer(
            "Results",
            [
                self.results_dropdown,
                self.view_results_type_button,
            ],
        ).render()


class CorrelationSection:
    def __init__(self, config):
        self.correlation_type_dropdown = dcc.Dropdown(
            id="correlation_coefficient",
            multi=False,
            placeholder="Select Coefficient...",
            options=generate_dropdown_options(
                config.get("coefficients", ["Wrong Columns"])
            ),  # Assuming you have a function to generate options
            style={
                "width": "100%"
            },  # Assuming you want to use the full width for styling
        )

        self.plot_button = CustomButton(
            "Calculate Correlation", "calculate_correlation"
        ).render()

    def render(self):
        return SectionContainer(
            "Correlation Analysis",
            [
                self.correlation_type_dropdown,
                self.plot_button,  # Include the message
            ],
        ).render()


class CustomAnalysisSection:
    def __init__(self, config):
        self.custom_distribution_dropdown = dcc.Dropdown(
            id="custom_analysis_type",
            multi=False,
            placeholder="Select Analysis...",
            options=generate_dropdown_options(
                config.get("custom_analysis", ["Wrong Columns"])
            ),  # Assuming you have a function to generate options
            style={
                "width": "100%"
            },  # Assuming you want to use the full width for styling
        )

        self.plot_button = CustomButton(
            "Plot Custom Analysis", "plot_custom_analysis"
        ).render()

    def render(self):
        return SectionContainer(
            "Custom Analysis",
            [
                self.custom_distribution_dropdown,
                self.plot_button,  # Include the message
            ],
        ).render()


class DatasetTabLayout:
    def __init__(self, config_manager):
        self.variants = (
            config_manager.variants
        )  # You might want to use config settings if applicable
        self.dataset_tab_config = (
            config_manager.dataset_tab
        )  # You might want to use config settings if applicable

    def render(self):

        select_variant_container = VariantSection(self.variants).render()
        #  Summary Statistics Section Container
        summary_container = SummarySection(
            self.dataset_tab_config
        ).render()  # Create and render the distribution section
        # summary_statistics_container = html.Div(
        #     id="summary_container",
        #     style={'width': '50%', 'height': '100%'}  # Initially hidden
        # )
        summary_statistics_container = html.Div(
            id="summary_container", style={"width": "50%", "height": "100%"}
        )

        # Distribution Section Container
        distribution_container = DistributionSection(
            self.dataset_tab_config
        ).render()  # Create and render the distribution section

        # distribution_graph_container = LoadingContainer(
        #     container_id="distribution_container",
        #     loader_id="distribution_graph_loader"
        # ).render()
        distribution_graph_container = LoadingContainer(
            container_id="distribution_container",
            loader_id="distribution_graph_loader",
            container_style={"width": "70%", "height": "500px"},
        ).render()

        results_container = ResultsSection(self.dataset_tab_config).render()

        # results_data_table = html.Div(
        #     id="results_output_container",
        #     style={'width': '50%', 'height': '100%'}  # Initially hidden
        # )
        results_data_table = html.Div(
            id="results_output_container", style={"width": "50%", "height": "100%"}
        )

        correlation_container = CorrelationSection(self.dataset_tab_config).render()

        # correlation_matrix_container = LoadingContainer(
        #     container_style={
        #         'width': '45%',  # Consistent width across all containers
        #         'height': '600px',  # Consistent height across all containers
        #         'display': 'inline-block',  # Display inline to allow for side-by-side alignment
        #         'vertical-align': 'top',
        #         'padding': '10px'
        #     },

        #     container_id="correlation_matrix_container",
        #     loader_id="correlation_matrix_loader"
        # ).render()

        # correlation_scatter_container = LoadingContainer(
        #     container_style={
        #         'width': '45%',  # Consistent width across all containers
        #         'height': '600px',  # Consistent height across all containers
        #         'display': 'inline-block',  # Display inline to allow for side-by-side alignment
        #         'vertical-align': 'top',
        #         'padding': '10px'
        #     },
        #     container_id="correlation_scatter_container",
        #     loader_id="correlation_scatter_loader"
        # ).render()

        # correlation_graphs = html.Div(
        #     [
        #         correlation_matrix_container,
        #         correlation_scatter_container
        #     ],
        #     style={
        #         'display': 'flex',
        #         'flex-direction': 'row',
        #         'width': '100%',
        #         'justify-content': 'space-around'  # Space evenly between graphs
        #     }
        # )

        # correlation_graphs = html.Div(
        #     children=[
        #         LoadingContainer(
        #             container_id="correlation_matrix_container",
        #             loader_id="correlation_matrix_loader",
        #             container_style={'width': '49%', 'height': '600px'}
        #         ).render(),
        #         LoadingContainer(
        #             container_id="correlation_scatter_container",
        #             loader_id="correlation_scatter_loader",
        #             container_style={'width': '49%', 'height': '600px'}
        #         ).render()
        #     ],
        #     className='graph-container'
        # )
        # Import required modules

        # Modify your layout code
        # correlation_graphs = html.Div(
        #     children=[
        #         # Placeholder for correlation matrix graph
        #         dcc.Graph(
        #             id="correlation_matrix_graph",
        #             figure={},  # Empty figure as a placeholder
        #             style={'width': '49%', 'display': 'inline-block'}
        #         ),
        #         # Placeholder for scatter plot graph
        #         dcc.Graph(
        #             id="correlation_scatter_graph",
        #             figure={},  # Empty figure as a placeholder
        #             style={'width': '49%', 'display': 'inline-block'}
        #         )
        #     ],
        #     className='graph-container'
        # )
        correlation_graphs = html.Div(
            children=[
                # Loading component for correlation matrix graph
                dcc.Loading(
                    id="loading_correlation_matrix",
                    type="default",
                    children=[
                        dcc.Graph(
                            id="correlation_matrix_graph",
                            figure={},  # Empty figure as a placeholder
                            style={"width": "49%", "display": "none"},
                        )
                    ],
                    style={"display": "inline-block", "width": "49%"},
                ),
                # Loading component for scatter plot graph
                dcc.Loading(
                    id="loading_correlation_scatter",
                    type="default",
                    children=[
                        dcc.Graph(
                            id="correlation_scatter_graph",
                            figure={},  # Empty figure as a placeholder
                            style={"width": "49%", "display": "none"},
                        )
                    ],
                    style={"display": "inline-block", "width": "49%"},
                ),
            ],
            className="graph-container",
        )

        custom_analysis_container = CustomAnalysisSection(
            self.dataset_tab_config
        ).render()

        # custom_analysis_graph_container = LoadingContainer(
        #     container_id="custom_analysis_graph_container",
        #     loader_id="custom_analysis_graph_loader",
        #     # container_style={'width': '50%', 'height': '500px'}
        #     container_style={'width': '50%', 'height': '500px', 'overflow': 'hidden' }
        # ).render()
        custom_analysis_graph_container = LoadingContainer(
            container_id="custom_analysis_graph_container",
            loader_id="custom_analysis_graph_loader",
            container_style={
                "width": "50%",  # Adjust width as needed
                "height": "500px",  # Fixed height
                "margin": "auto",  # Center the container
                "overflow": "hidden",  # Hide any overflow outside the container
            },
            inner_container_style={
                "width": "100%",  # Full width
                "height": "100%",  # Full height
                "overflowX": "auto",  # Enable horizontal scrolling
                "overflowY": "hidden",  # Disable vertical scrolling
                "whiteSpace": "nowrap",  # Prevent line breaks in the table content
            },
        ).render()

        layout = html.Div(
            # className='main-container',
            className="main-container center-container",
            children=[
                select_variant_container,
                summary_container,
                summary_statistics_container,
                distribution_container,
                distribution_graph_container,
                results_container,
                results_data_table,
                correlation_container,
                correlation_graphs,
                custom_analysis_container,
                custom_analysis_graph_container,
            ],
            # style={
            #     'display': 'flex',
            #     'flex-direction': 'column',
            #     'align-items': 'center',  # Center the content horizontally
            #     'justify-content': 'center',  # Center the content vertically
            #     'width': '100%',
            #     'padding': '20px',
            #     'box-sizing': 'border-box',
            # }
        )

        return layout
