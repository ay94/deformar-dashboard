
from layouts.managers.layout_managers import (CustomButton, DropdownContainer,
                                              LoadingContainer,
                                              SectionContainer, LanguageSelection,
                                              generate_dropdown_options)


from typing import Dict, List, Optional
from dash import html, dcc
import dash_bootstrap_components as dbc

class ComponentSection:
    """One component section: header + (analysis dd, figure dd) + canvas.
       Expects a dict like:
       {
         "id": "data",
         "label": "Data Component",
         "analyses": [
           {
             "id": "structural",
             "label": "Structural Analysis",
             "figures": [
               {"id": "dataset_stats", "label": "Dataset Stats", "plot": "table", "required_columns": None}
             ]
           }
         ]
       }
    """
    def __init__(self, component_cfg: Dict):
        self.cfg = component_cfg
        self.comp_id: str = component_cfg["id"]
        self.comp_label: str = component_cfg.get("label", self.comp_id.title())

    def _analysis_options(self) -> List[Dict[str, str]]:
        return [{"label": a["label"], "value": a["id"]} for a in self.cfg.get("analyses", [])]

    def _figure_options_for(self, analysis_id: Optional[str]) -> List[Dict[str, str]]:
        if not analysis_id:
            return []
        ana = next((a for a in self.cfg.get("analyses", []) if a["id"] == analysis_id), None)
        if not ana:
            return []
        return [{"label": f["label"], "value": f["id"]} for f in ana.get("figures", [])]

    def _first_analysis_id(self) -> Optional[str]:
        analyses = self.cfg.get("analyses", [])
        return analyses[0]["id"] if analyses else None

    def render(self) -> dbc.Container:
        # first_aid = self._first_analysis_id()

        analysis_dd = dcc.Dropdown(
            id={"type": "analysis-dd", "component": self.comp_id},
            options=self._analysis_options(),
            value=None,     # default to first analysis if present
            clearable=False,
            multi=False,
            placeholder="Select analysis...",
        )

        figure_dd = dcc.Dropdown(
            id={"type": "figure-dd", "component": self.comp_id},
            # options=self._figure_options_for(first_aid),
            options=[],
            value=None,
            clearable=True,
            multi=False,
            placeholder="Select figure...",
        )
        render_btn = CustomButton(
            "Render Canvas",
            {"type": "render-btn", "component": self.comp_id}
        ).render()

        canvas = dcc.Loading(
            id={"type": "canvas-loading", "component": self.comp_id},
            type="default",
            children=html.Div(
                id={"type": "figure-canvas", "component": self.comp_id},
                style={"minHeight": "320px"},
            ),
        )

           # Left column: header + stacked dropdowns (full width inside the column)
        controls_col = dbc.Col(
            [
                
                analysis_dd,
                
                html.Div(style={"height": "10px"}),  # small spacer

                figure_dd,

                render_btn
            ],
            md=4,  # sidebar width; adjust if you want slimmer/wider
            style={"maxWidth": "420px"}  # keep menu from feeling 'full screen'
        )

        # Right column: full canvas area
        canvas_col = dbc.Col(
            canvas,
            md=8,
        )

       

        return dbc.Container(
            [
                dbc.Row(
                    [
                        html.H2(self.comp_label, className="section-header"),
                        controls_col,
                        canvas_col
                    ],
                    align="start",
                    style={"marginTop": "8px", "marginBottom": "16px"},
                ),
                html.Hr(),
            ],
            fluid=True,
        )



class CrossComponentTab:
    """Builds the entire tab from the YAML dict at cross_component_tab."""
    def __init__(self, config_manager):
        # keep your existing pattern
        self.variants = config_manager.variants
    
      
        self.components_cfg: List[Dict] = config_manager.cross_component.get("components", [])
      
    def render(self):
        # language selector stays as you had it
        select_language_container = LanguageSelection(self.variants + ["combined"]).render()
        # Build a section per component present in config
        sections = []
        if not self.components_cfg:
            sections.append(html.Div("No components found in cross_component_tab.", className="text-danger text-center"))
        else:
            for comp_cfg in self.components_cfg:
                sections.append(ComponentSection(comp_cfg).render())

        return html.Div(
            className="main-container center-container",
            children=[select_language_container, *sections],
        )

        # # component section for DATA (header + 2 dropdowns + canvas)
        # data_section = ComponentSection(self.data_component_cfg).render() if self.data_component_cfg else html.Div(
        #     "No 'data' component found in cross_component_tab.", className="text-danger text-center"
        # )

        # # page layout (you can swap Container/Card/etc. – structure stays the same)
        # layout = html.Div(
        #     className="main-container center-container",
        #     children=[
        #         select_language_container,
        #         data_section,
        #     ],
        # )
        # return layout



# class DatasetTabLayout:
#     def __init__(self, config_manager):
#         print(config_manager.variants)
#         self.variants = (
#             config_manager.variants
#         )  # You might want to use config settings if applicable
#         self.dataset_tab_config = (
#             config_manager.quantitative
#         )  # You might want to use config settings if applicable

#     def render(self):

#         select_language_container = LanguageSelection(self.variants+['combined']).render()

#         #TODO: Define Section for the dataset Component. 
#         #TODO: Specify a dropdown for each type of analysis. 
#         #TODO: Specify a dropdown for the analysis graph itself. 
        
        
#         summary_container = SummarySection(
#             self.dataset_tab_config
#         ).render()  # Create and render the distribution section
       
#         layout = html.Div(
#             # className='main-container',
#             className="main-container center-container",
#             children=[
#                 select_language_container,
#             ],
#         )

#         return layout











# class SummarySection:
#     def __init__(self, config):
#         self.statistical_dropdown = dcc.Dropdown(
#             id="statistical_columns",
#             multi=True,
#             placeholder="Select column...",
#             options=generate_dropdown_options(
#                 config.get("statistics_columns", ["Wrong Columns"])
#             ),  # Assuming you have a function to generate options
#         )
#         self.generate_summary_button = CustomButton(
#             "Generate Summary Statistics", "generate_summary"
#         ).render()

#     def render(self):

#         return SectionContainer(
#             "Summary Statistics",
#             [
#                 self.statistical_dropdown,
#                 self.generate_summary_button,
#             ],
#         ).render()


# class DistributionSection:
#     def __init__(self, config):
#         self.distribution_column_dropdown = dcc.Dropdown(
#             id="distribution_column",
#             multi=False,
#             placeholder="Select Distribution column...",
#             options=generate_dropdown_options(
#                 config.get("statistics_columns", ["Wrong Columns"])
#             ),  # Assuming you have a function to generate options
#             style={
#                 "width": "100%"
#             },  # Assuming you want to use the full width for styling
#         )
#         self.categorical_column_dropdown = dcc.Dropdown(
#             id="categorical_column",
#             multi=False,
#             placeholder="Select Categorical column...",
#             options=generate_dropdown_options(
#                 config.get("categorical_columns", ["Wrong Columns"])
#             ),  # Assuming you have a function to generate options
#             style={"width": "100%"},
#         )
#         self.plot_button = CustomButton(
#             "Plot Distribution", "plot_distribution"
#         ).render()

#     def render(self):
#         return SectionContainer(
#             "Distribution Analysis",
#             [
#                 self.distribution_column_dropdown,
#                 self.categorical_column_dropdown,
#                 self.plot_button,  # Include the message
#             ],
#         ).render()


# class ResultsSection:
#     def __init__(self, config):
#         self.results_dropdown = dcc.Dropdown(
#             id="results_type",
#             multi=False,
#             placeholder="Select results...",
#             options=generate_dropdown_options(
#                 config.get("results", [])
#             ),  # Assuming you have a function to generate options
#         )
#         self.view_results_type_button = CustomButton(
#             "View Results", "view_results_type"
#         ).render()

#     def render(self):

#         return SectionContainer(
#             "Results",
#             [
#                 self.results_dropdown,
#                 self.view_results_type_button,
#             ],
#         ).render()


# class CorrelationSection:
#     def __init__(self, config):
#         self.correlation_type_dropdown = dcc.Dropdown(
#             id="correlation_coefficient",
#             multi=False,
#             placeholder="Select Coefficient...",
#             options=generate_dropdown_options(
#                 config.get("coefficients", ["Wrong Columns"])
#             ),  # Assuming you have a function to generate options
#             style={
#                 "width": "100%"
#             },  # Assuming you want to use the full width for styling
#         )
#         self.categorical_column_dropdown = dcc.Dropdown(
#             id="correlation_categorical_column",
#             multi=False,
#             placeholder="Select Categorical column...",
#             options=generate_dropdown_options(
#                 config.get("categorical_columns", ["Wrong Columns"])
#             ),  # Assuming you have a function to generate options
#             style={"width": "100%"},
#         )
#         self.plot_button = CustomButton(
#             "Calculate Correlation", "calculate_correlation"
#         ).render()

#     def render(self):
#         return SectionContainer(
#             "Correlation Analysis",
#             [
#                 self.correlation_type_dropdown,
#                 self.categorical_column_dropdown,
#                 self.plot_button,  # Include the message
#             ],
#         ).render()


# class CustomAnalysisSection:
#     def __init__(self, config):
#         self.custom_distribution_dropdown = dcc.Dropdown(
#             id="custom_analysis_type",
#             multi=False,
#             placeholder="Select Analysis...",
#             options=generate_dropdown_options(
#                 config.get("custom_analysis", ["Wrong Columns"])
#             ),  # Assuming you have a function to generate options
#             style={
#                 "width": "100%"
#             },  # Assuming you want to use the full width for styling
#         )

#         self.plot_button = CustomButton(
#             "Plot Custom Analysis", "plot_custom_analysis"
#         ).render()

#     def render(self):
#         return SectionContainer(
#             "Custom Analysis",
#             [
#                 self.custom_distribution_dropdown,
#                 self.plot_button,  # Include the message
#             ],
#         ).render()





# class DatasetTabLayout:
#     def __init__(self, config_manager):
#         print(config_manager.variants)
#         self.variants = (
#             config_manager.variants
#         )  # You might want to use config settings if applicable
#         self.dataset_tab_config = (
#             config_manager.quantitative
#         )  # You might want to use config settings if applicable

#     def render(self):

#         select_language_container = LanguageSelection(self.variants+['combined']).render()
        
#         summary_container = SummarySection(
#             self.dataset_tab_config
#         ).render()  # Create and render the distribution section
#         # summary_statistics_container = html.Div(
#         #     id="summary_container",
#         #     style={'width': '50%', 'height': '100%'}  # Initially hidden
#         # )
#         summary_statistics_container = html.Div(
#             id="summary_container", style={"width": "50%", "height": "100%"}
#         )

#         # Distribution Section Container
#         distribution_container = DistributionSection(
#             self.dataset_tab_config
#         ).render()  # Create and render the distribution section

#         # distribution_graph_container = LoadingContainer(
#         #     container_id="distribution_container",
#         #     loader_id="distribution_graph_loader"
#         # ).render()
#         distribution_graph_container = LoadingContainer(
#             container_id="distribution_container",
#             loader_id="distribution_graph_loader",
#             container_style={"width": "70%", "height": "500px"},
#         ).render()

#         results_container = ResultsSection(self.dataset_tab_config).render()

#         # results_data_table = html.Div(
#         #     id="results_output_container",
#         #     style={'width': '50%', 'height': '100%'}  # Initially hidden
#         # )
#         results_data_table = html.Div(
#             id="results_output_container", style={"width": "50%", "height": "100%"}
#         )

#         correlation_container = CorrelationSection(self.dataset_tab_config).render()

#         # correlation_matrix_container = LoadingContainer(
#         #     container_style={
#         #         'width': '45%',  # Consistent width across all containers
#         #         'height': '600px',  # Consistent height across all containers
#         #         'display': 'inline-block',  # Display inline to allow for side-by-side alignment
#         #         'vertical-align': 'top',
#         #         'padding': '10px'
#         #     },

#         #     container_id="correlation_matrix_container",
#         #     loader_id="correlation_matrix_loader"
#         # ).render()

#         # correlation_scatter_container = LoadingContainer(
#         #     container_style={
#         #         'width': '45%',  # Consistent width across all containers
#         #         'height': '600px',  # Consistent height across all containers
#         #         'display': 'inline-block',  # Display inline to allow for side-by-side alignment
#         #         'vertical-align': 'top',
#         #         'padding': '10px'
#         #     },
#         #     container_id="correlation_scatter_container",
#         #     loader_id="correlation_scatter_loader"
#         # ).render()

#         # correlation_graphs = html.Div(
#         #     [
#         #         correlation_matrix_container,
#         #         correlation_scatter_container
#         #     ],
#         #     style={
#         #         'display': 'flex',
#         #         'flex-direction': 'row',
#         #         'width': '100%',
#         #         'justify-content': 'space-around'  # Space evenly between graphs
#         #     }
#         # )

#         # correlation_graphs = html.Div(
#         #     children=[
#         #         LoadingContainer(
#         #             container_id="correlation_matrix_container",
#         #             loader_id="correlation_matrix_loader",
#         #             container_style={'width': '49%', 'height': '600px'}
#         #         ).render(),
#         #         LoadingContainer(
#         #             container_id="correlation_scatter_container",
#         #             loader_id="correlation_scatter_loader",
#         #             container_style={'width': '49%', 'height': '600px'}
#         #         ).render()
#         #     ],
#         #     className='graph-container'
#         # )
#         # Import required modules

#         # Modify your layout code
#         # correlation_graphs = html.Div(
#         #     children=[
#         #         # Placeholder for correlation matrix graph
#         #         dcc.Graph(
#         #             id="correlation_matrix_graph",
#         #             figure={},  # Empty figure as a placeholder
#         #             style={'width': '49%', 'display': 'inline-block'}
#         #         ),
#         #         # Placeholder for scatter plot graph
#         #         dcc.Graph(
#         #             id="correlation_scatter_graph",
#         #             figure={},  # Empty figure as a placeholder
#         #             style={'width': '49%', 'display': 'inline-block'}
#         #         )
#         #     ],
#         #     className='graph-container'
#         # )
#         correlation_graphs = html.Div(
#             children=[
#                 # Loading component for correlation matrix graph
#                 dcc.Loading(
#                     id="loading_correlation_matrix",
#                     type="default",
#                     children=[
#                         dcc.Graph(
#                             id="correlation_matrix_graph",
#                             figure={},  # Empty figure as a placeholder
#                             style={"width": "49%", "display": "none"},
#                         )
#                     ],
#                     style={"display": "inline-block", "width": "49%"},
#                 ),
#                 # Loading component for scatter plot graph
#                 dcc.Loading(
#                     id="loading_correlation_scatter",
#                     type="default",
#                     children=[
#                         dcc.Graph(
#                             id="correlation_scatter_graph",
#                             figure={},  # Empty figure as a placeholder
#                             style={"width": "49%", "display": "none"},
#                         )
#                     ],
#                     style={"display": "inline-block", "width": "49%"},
#                 ),
#             ],
#             className="graph-container",
#         )
#         correlation_prompt = html.Div(id="correlation_prompt")

#         custom_analysis_container = CustomAnalysisSection(
#             self.dataset_tab_config
#         ).render()

#         # custom_analysis_graph_container = LoadingContainer(
#         #     container_id="custom_analysis_graph_container",
#         #     loader_id="custom_analysis_graph_loader",
#         #     # container_style={'width': '50%', 'height': '500px'}
#         #     container_style={'width': '50%', 'height': '500px', 'overflow': 'hidden' }
#         # ).render()
#         custom_analysis_graph_container = LoadingContainer(
#             container_id="custom_analysis_graph_container",
#             loader_id="custom_analysis_graph_loader",
#             container_style={
#                 "width": "50%",  # Adjust width as needed
#                 "height": "500px",  # Fixed height
#                 "margin": "auto",  # Center the container
#                 "overflow": "hidden",  # Hide any overflow outside the container
#             },
#             inner_container_style={
#                 "width": "100%",  # Full width
#                 "height": "100%",  # Full height
#                 "overflowX": "auto",  # Enable horizontal scrolling
#                 "overflowY": "hidden",  # Disable vertical scrolling
#                 "whiteSpace": "nowrap",  # Prevent line breaks in the table content
#             },
#         ).render()

#         layout = html.Div(
#             # className='main-container',
#             className="main-container center-container",
#             children=[
#                 select_language_container,
#                 # summary_container,
#                 # summary_statistics_container,
#                 # distribution_container,
#                 # distribution_graph_container,
#                 # correlation_container,
#                 # correlation_graphs,
#                 # correlation_prompt,
#                 # custom_analysis_container,
#                 # custom_analysis_graph_container,
#                 # results_container,
#                 # results_data_table,
#             ],
#             # style={
#             #     'display': 'flex',
#             #     'flex-direction': 'column',
#             #     'align-items': 'center',  # Center the content horizontally
#             #     'justify-content': 'center',  # Center the content vertically
#             #     'width': '100%',
#             #     'padding': '20px',
#             #     'box-sizing': 'border-box',
#             # }
#         )

#         return layout
