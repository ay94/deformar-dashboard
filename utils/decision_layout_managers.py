from dash import html, dcc
from dash import html
from utils.layout_managers import (
    CustomButton,
    generate_dropdown_options,
    DropdownContainer,
    SectionContainer,
    LoadingContainer, 
    VariantSection
)
import plotly.graph_objs as go

class DecisionSection:
    def __init__(self, config):
        self.decision_type_dropdown = dcc.Dropdown(
            id='decision_type',
            multi=False,
            placeholder="Select Decision type...",
            options=generate_dropdown_options(config.get('decision_type', ['Wrong Columns'])),  # Assuming you have a function to generate options
            style={'width': '100%'}  # Assuming you want to use the full width for styling
        )
        
        self.decision_columns_dropdown = dcc.Dropdown(
            id='decision_columns',
            multi=False,
            placeholder="Select Color column...",
            options=generate_dropdown_options(config.get('decision_columns', ['Wrong Columns'])),  # Assuming you have a function to generate options
            style={'width': '100%'}  # Assuming you want to use the full width for styling
        )
        self.decision_correlation_type_dropdown = dcc.Dropdown(
            id='decision_correlation_coefficient',
            multi=False,
            placeholder="Select Coefficient...",
            options=generate_dropdown_options(config.get('coefficients', ['Wrong Columns'])),  # Assuming you have a function to generate options
            style={'width': '100%'}  # Assuming you want to use the full width for styling
        )
        
        self.plot_button = CustomButton('View Decision Boundary', 'view_decision_boundary').render()
        
    def render(self):
        return SectionContainer(
                    "Decision Boundary Analysis", 
                    [
                        self.decision_type_dropdown,
                        self.decision_columns_dropdown, 
                        self.decision_correlation_type_dropdown,
                        self.plot_button  # Include the message
                    ]
                ).render()

# html.Div([
#         # Left container for the heatmap
        
#         dcc.Loading(
#             id="loading_heatmap",
#             type="default",
#             children=[
#                 dcc.Graph(
#                     id="heatmap_graph",
#                     figure=go.Figure(),  # Placeholder until data is loaded
#                     style={'width': '49%', 'height': '100%'}
#                 )
#             ],
#             style={'display': 'inline-block', 'width': '49%', 'height': '100%'}
#         ),
#         # Right container for scatter plots
#         html.Div([
#             dcc.Loading(
#                 id="loading_scatter_top",
#                 type="default",
#                 children=[
#                     dcc.Graph(
#                         id="scatter_top_graph",
#                         figure=go.Figure(),  # Placeholder until data is loaded
#                     )
#                 ],
#                 style={'width': '100%', 'height': '50%'}  # Takes half of the right column
#             ),
#             dcc.Loading(
#                 id="loading_scatter_bottom",
#                 type="default",
#                 children=[
#                     dcc.Graph(
#                         id="scatter_bottom_graph",
#                         figure=go.Figure(),  # Placeholder until data is loaded
#                     )
#                 ],
#                 style={'width': '100%', 'height': '50%'}  # Takes the remaining half of the right column
#             )
#         ], style={'display': 'inline-block', 'width': '49%', 'height': '100%', 'verticalAlign': 'top'})
#     ], 
#         className='graph-container'
# )






class DecisionTabLayout:
    def __init__(self, config_manager):
        self.variants = config_manager.variants  # You might want to use config settings if applicable
        self.decision_tab_config = config_manager.decision_tab  # You might want to use config settings if applicable

    def render(self):
        
        select_variant_container = VariantSection(self.variants).render()
        
        training_header = html.H3("Training Boundary", className='section-header')
        training_graph_container = LoadingContainer(
            container_id="training_graph",
            loader_id="training_graph_loader",
            container_style={'width': '70%', 'height': '500px'}
        ).render()
       
        
        decision_container = DecisionSection(self.decision_tab_config).render()  # Create and render the distribution section
        
        decision_graph = dcc.Loading(
            id="loading_decision_graph",
            type="default",
            children=[
                dcc.Graph(
                    id="decision_boundary_graph",
                    figure={},  # Empty figure as a placeholder
                    style={'width': '100%', 'display': 'none'}
                )
            ],
            style={'display': 'inline-block', 'width': '100%'}
        )
        decision_graph_prompt = html.Div(id='decision_graph_prompt')
        # decision_clusters = html.Div(
        #     children=[
        #         # Left container for the heatmap
        #         dcc.Loading(
        #             id="loading_correlation_heatmap",
        #             type="default",
        #             children=[
        #                 dcc.Graph(
        #                     id="correlation_heatmap",
        #                     figure={},  # Placeholder until data is loaded
        #                     style={'height': '100%', 'width': '100%', 'display': 'none'}
        #                 )
        #             ],
        #             style={'display': 'inline-block', 'width': '30%', 'height': '100%'}
        #         ),
        #         # Right container for scatter plots
        #         html.Div(
        #             children=[
        #                 dcc.Loading(
        #                     id="loading_decision_scatter",
        #                     type="default",
        #                     children=[
        #                         dcc.Graph(
        #                             id="decision_scatter",
        #                             figure=go.Figure(),  # Placeholder until data is loaded
        #                             style={'width': '100%', 'height': '100%'}
        #                         )
        #                     ],
        #                     style={'width': '100%', 'height': '50%'}  # Takes half of the right column
        #                 ),
        #                 dcc.Loading(
        #                     id="loading_measure_scatter",
        #                     type="default",
        #                     children=[
        #                         dcc.Graph(
        #                             id="measure_scatter",
        #                             figure=go.Figure(),  # Placeholder until data is loaded
        #                             style={'width': '100%', 'height': '100%'}
        #                         )
        #                     ],
        #                     style={'width': '100%', 'height': '50%'}  # Takes the remaining half of the right column
        #                 )
        #             ], 
        #             style={'display': 'inline-block', 'width': '70%', 'height': '100%'})
        #     ], 
        #     style={'display': 'flex', 'height': '90vh'} # Ensures the container takes full height
        # )  
#         decision_clusters = html.Div(
#     children=[
#         # Left container for the heatmap
#         dcc.Loading(
#             id="loading_correlation_heatmap",
#             type="default",
#             children=[
#                 dcc.Graph(
#                     id="correlation_heatmap",
#                     figure={},  # Placeholder until data is loaded
#                     className='graph-full'  # Using CSS for styling
#                 )
#             ],
#             className='heatmap-container'
#         ),
#         # Right container for scatter plots
#         html.Div(
#             children=[
#                 dcc.Loading(
#                     id="loading_decision_scatter",
#                     type="default",
#                     children=[
#                         dcc.Graph(
#                             id="decision_scatter",
#                             figure=go.Figure(),  # Placeholder until data is loaded
#                             className='graph-full'
#                         )
#                     ],
#                     className='scatter-plot'
#                 ),
#                 # dcc.Loading(
#                 #     id="loading_measure_scatter",
#                 #     type="default",
#                 #     children=[
#                 #         dcc.Graph(
#                 #             id="measure_scatter",
#                 #             figure=go.Figure(),  # Placeholder until data is loaded
#                 #             className='graph-full'
#                 #         )
#                 #     ],
#                 #     className='scatter-plot'
#                 # )
#             ], 
#             className='scatter-plots-container'
#         )
#     ], 
#     className='decision-clusters'  # Main container class
# )
        
#         graph = dcc.Loading(
#                     id="loading_correlation_scatter",
#                     type="default",
#                     children=[
#                         dcc.Graph(
#                             id="gdf",
#                             figure=go.Figure(),  # Empty figure as a placeholder
#                             style={'width': '100%'}
#                         )
#                     ],
#                     style={'display': 'inline-block', 'width': '100%'}
#                 )
        decision_clusters = html.Div(
            children=[
                # First row with heatmap and first scatter plot
                html.Div(
                    children=[
                        # Left container for the heatmap
                        dcc.Loading(
                            id="loading_correlation_heatmap",
                            type="default",
                            children=[
                                dcc.Graph(
                                    id="correlation_heatmap",
                                    figure={},  # Placeholder until data is loaded
                                    className='graph-full'
                                )
                            ],
                            className='heatmap-container'
                        ),
                        # Right container for first scatter plot
                        dcc.Loading(
                            id="loading_decision_scatter",
                            type="default",
                            children=[
                                dcc.Graph(
                                    id="measure_scatter",
                                    figure=go.Figure(),  # Placeholder until data is loaded
                                    className='graph-full'
                                )
                            ],
                            className='scatter-plot'
                        )
                    ],
                    className='top-row-container'  # New class for top row
                ),
                # Second row with second scatter plot full width
                dcc.Loading(
                    id="loading_measure_scatter",
                    type="default",
                    children=[
                        dcc.Graph(
                            id="decision_scatter",
                            figure=go.Figure(),  # Placeholder until data is loaded
                            className='graph-full'
                        )
                    ],
                    className='full-width-scatter'  # New class for full width scatter
                )
            ],
            className='decision-clusters'
        )

        layout = html.Div(
                    # className='main-container',
                    className='main-container center-container',
                    children=[
                        select_variant_container,
                        training_header,
                        training_graph_container,
                        decision_container,
                        decision_graph,
                        decision_graph_prompt,
                        decision_clusters, 
                        
                    ],
                    
                    
            )

        return layout




