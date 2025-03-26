import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import dcc, html

from layouts.managers.layout_managers import (CustomButton, LoadingContainer,
                                              SectionContainer, VariantSection,
                                              FilterLayerSection,
                                              generate_dropdown_options)
from config.enums import (CorrelationCoefficients, DecisionType,
                          DisplayColumns,
                          SelectionPlotColumns)

from dash import dash_table
class FilterLayer:
    def __init__(self, config):
        self.filter_column_dropdown = dcc.Dropdown(
            id="filter_column_dropdown",
            placeholder="Select Category...",
            options=generate_dropdown_options(
                config.get("categorical_columns", ["Wrong Columns"])
            ),
            style={"minWidth": "180px", "marginRight": "10px"},
        )
        self.filter_value_dropdown = dcc.Dropdown(
            id="filter_value_dropdown",
            placeholder="Select Value...",
            style={"minWidth": "180px", "marginRight": "10px"},
        )
        self.apply_button = CustomButton("Apply Filter", "filter_table_button").render()
        self.reset_button = CustomButton("Reset Filter", "reset_filter_button").render()
        self.data_table = dash_table.DataTable(
            id='filtered_data_table',
            columns=[],  # will be filled by callback
            data=[],     # will be filled by callback
            editable=True,
            filter_action="native",
            sort_action="native",
            sort_mode='multi',
            column_selectable="single",
            row_selectable='multi',
            row_deletable=True,
            selected_rows=[],
            page_action='native',
            page_current=0,
            page_size=10,
            style_header={
                'text-align': "center",
                'background-color': "#4bb3a8",
                'color': 'white',
                "fontWeight": "bold",
            },
        )
        self.filtered = dcc.Store(id="filter_state", data={"filtered": False})

    def render(self):
        # Filters row
        filter_controls = html.Div(
            children=[
                self.apply_button,
                self.filter_column_dropdown,
                self.filter_value_dropdown,
                self.reset_button,
            ],
            style={
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center",
                "gap": "10px",
                "flexWrap": "wrap",
                "marginBottom": "20px",
            },
        )

        # Table below
        data_table_wrapper = html.Div(
            children=[
                dcc.Loading(
                    id="loading_decision_container",
                    type="default",
                    children=[
                        self.data_table
                    ],
                ),
                self.filtered,
            ],
            style={
                "width": "95%",
                "display": "flex",
                "justifyContent": "center",
            },
        )

        return FilterLayerSection(
            'Filtering Layer',
            content_components=[
                filter_controls,
                data_table_wrapper,
            ]
        ).render()
       
class DecisionSection:
    def __init__(self, config):
        
        self.model_type_dropdown = dcc.Dropdown(
            id="model_type",
            multi=False,
            placeholder="Select Model type...",
            options=generate_dropdown_options(
                config.get("model_type", ["Wrong Columns"])
            ),  # Assuming you have a function to generate options
            style={
                "width": "100%"
            },  # Assuming you want to use the full width for styling
        )

        self.decision_columns_dropdown = dcc.Dropdown(
            id="decision_columns",
            multi=True,
            placeholder="Select Decision Color column...",
            options=generate_dropdown_options(
                config.get("categorical_columns", ["Wrong Columns"])
            ),  # Assuming you have a function to generate options
            style={
                "width": "100%"
            },  # Assuming you want to use the full width for styling
        )
        self.measure_columns_dropdown = dcc.Dropdown(
            id="measure_columns",
            multi=True,
            placeholder="Select Measure Color column...",
            options=generate_dropdown_options(
                config.get("categorical_columns", ["Wrong Columns"])
            ),  # Assuming you have a function to generate options
            style={
                "width": "100%"
            },  # Assuming you want to use the full width for styling
        )
        self.correlation_columns_dropdown = dcc.Dropdown(
            id="correlation_columns",
            multi=True,
            placeholder="Select Correlation columns...",
            options=generate_dropdown_options(
                config.get("numerical_columns", ["Wrong Columns"])
            ),  # Assuming you have a function to generate options
            style={
                "width": "100%"
            },  # Assuming you want to use the full width for styling
        )
        self.decision_correlation_type_dropdown = dcc.Dropdown(
            id="decision_correlation_coefficient",
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
            "View Decision Boundary", "view_decision_boundary"
        ).render()
        self.clear_decision = CustomButton("Clear Decision", "clear_decision").render()
        self.clear_meeasure = CustomButton("Clear Measure", "clear_measure").render()

    def render(self):
        return SectionContainer(
            "Decision Boundary Analysis",
            [
                self.model_type_dropdown,
                self.measure_columns_dropdown,
                self.decision_columns_dropdown,
                self.correlation_columns_dropdown,
                self.decision_correlation_type_dropdown,
                self.plot_button,  # Include the message
                self.clear_decision,
                self.clear_meeasure,
            ],
        ).render()


class DecisionTabLayout:
    def __init__(self, config_manager):
        
        self.variants = (
            config_manager.variants
        )  # You might want to use config settings if applicable
        
        self.qualitative_tab_config = (
            config_manager.qualitative
        )  # You might want to use config settings if applicable

    def render(self):

        select_variant_container = VariantSection(self.variants).render()
        
        # Training Impact  
        training_header = html.H3("Training Boundary", className="section-header")
        
        view_train_decision = CustomButton(
            "View Training Decision Boundary", "view_training_decision_boundary"
        ).render()
        
        training_graph_container = LoadingContainer(
            container_id="training_graph",
            loader_id="training_graph_loader",
            container_style={"width": "70%", "height": "100%"},
        ).render()
        
        #  Filtering Layer 
        filter_layer_section = FilterLayer(
            self.qualitative_tab_config
            ).render()
        
        # Decision Layer 
        decision_section = DecisionSection(
            self.qualitative_tab_config
        ).render()  # Create and render the distribution section

        decision_graph_prompt = html.Div(id="decision_graph_prompt")

        measure_container = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Loading(
                                id="loading_heatmap_container",
                                type="default",
                                children=[
                                    dcc.Graph(id="correlation_heatmap", figure={})
                                ],
                            ),
                            width=5,
                        ),
                        dbc.Col(
                            dcc.Loading(
                                id="loading_measure_container",
                                type="default",
                                children=[
                                    dcc.Graph(id="measure_scatter", figure={}),
                                    # dcc.Store(id="measure_store"),
                                ],
                            ),
                            width=7,
                        ),
                    ],
                    style={"margin-bottom": "70px"},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Loading(
                                id="loading_decision_container",
                                type="default",
                                children=[
                                    dcc.Graph(id="decision_scatter", figure={}),
                                    # dcc.Store(id="decision_store"),
                                ],
                            ),
                            width=12,
                        )
                    ],
                    style={"margin-top": "50px"},
                ),
            ],
            fluid=True,
        )
        clustering_alignment_header = dbc.Container(
            dbc.Row(
                dbc.Col(
                    html.H4("Clustering Analysis", className="section-header"),
                    width=12,
                    style={
                        "text-align": "center",
                        "margin-top": "20px",
                        "margin-bottom": "20px",
                    },
                )
            ),
            fluid=True,
        )
        clustering_analysis = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Loading(
                                id="loading_clustering_results",
                                type="default",
                                children=html.Div(
                                    id="cluster_results_container",
                                    style={
                                        "width": "100%",
                                        "height": "auto",
                                        "overflow": "visible",
                                    },  # Use dynamic height and ensure overflow is visible
                                ),
                            ),
                            width=6,
                            style={"padding": "5px"},
                        ),
                        dbc.Col(
                            dcc.Loading(
                                id="loading_clustering_table",
                                type="default",
                                children=html.Div(
                                    id="centroid_matrix_container",
                                    style={
                                        "width": "100%",
                                        "height": "auto",
                                        "overflow": "visible",
                                    },  # Use dynamic height and ensure overflow is visible
                                ),
                            ),
                            width=6,
                            style={"padding": "5px"},
                        ),
                    ]
                )
            ],
            fluid=True,
        )
        
        selection_header = dbc.Container(
            dbc.Row(
                dbc.Col(
                    html.H4("Selections Summary", className="section-header"),
                    width=12,
                    style={
                        "text-align": "center",
                        "margin-top": "20px",
                        "margin-bottom": "20px",
                    },
                )
            ),
            fluid=True,
        )
        selection_analysis = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Dropdown(
                                        id="selection_tag_column",
                                        multi=False,
                                        placeholder="Select Category...",
                                        options=DisplayColumns.get_categorical(),
                                        # [],  # Assuming you have a function to generate options
                                        style={
                                            "width": "100%"
                                        },  # Assuming you want to use the full width for styling
                                    ),
                                dcc.Loading(
                                    id="loading_selection_tags",
                                    type="default",
                                    children=html.Div(
                                        id="selection_tag_container",
                                        style={
                                            "width": "100%",
                                            "height": "auto",
                                            "overflow": "visible",
                                        },  # Use dynamic height and ensure overflow is visible
                                    ),
                                    style={"marginTop": "30px"},  # 👈 add space between dropdown and content
                                ),
                            ],
                            width=6,
                            style={"padding": "5px"},
                        ),
                        dbc.Col(
                            [
                                dcc.Dropdown(
                                    id="selection_summary_column",
                                    multi=False,
                                    placeholder="Select Category...",
                                    options=DisplayColumns.get_categorical(),
                                    # [],  # Assuming you have a function to generate options
                                    style={
                                        "width": "100%"
                                    },  # Assuming you want to use the full width for styling
                                ),
                                dcc.Loading(
                                    id="loading_selection_summary",
                                    type="default",
                                    children=[
                                        html.Div(
                                            id="selection_summary_container",
                                            style={
                                                "width": "100%",
                                                "height": "auto",
                                                "overflow": "visible",
                                            },  # Use dynamic height and ensure overflow is visible
                                        ),  
                                        html.Div(
                                            id="selection_numeric_summary_container",
                                            style={
                                                "width": "100%",
                                                "height": "auto",
                                                "overflow": "visible",
                                            },  # Use dynamic height and ensure overflow is visible
                                        ),  
                                    ],
                                    style={"marginTop": "30px"},  # 👈 add space between dropdown and content
                                ),
                            ],
                            width=6,
                            style={"padding": "5px"},
                        ),
                    ]
                )
            ],
            fluid=True,
        )

        # Add a header between the two rows
        impact_header = dbc.Container(
            dbc.Row(
                dbc.Col(
                    html.H4("Training Impact Analysis", className="section-header"),
                    width=12,
                    style={
                        "text-align": "center",
                        "margin-top": "20px",
                        "margin-bottom": "20px",
                    },
                )
            ),
            fluid=True,
        )
        training_impact = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Loading(
                                id="loading_attention_similarity",
                                type="default",
                                children=html.Div(
                                    id="attention_similarity_matrix_container",
                                    style={
                                        "width": "100%",
                                        "height": "auto",
                                        "overflow": "visible",
                                    },  # Dynamic height and overflow settings
                                ),
                            ),
                            width=6,
                            style={
                                "padding": "50px",
                                "margin-bottom": "10px",
                            },  # Add bottom margin for spacing
                        ),
                        dbc.Col(
                            dcc.Loading(
                                id="loading_weight_similarity",
                                type="default",
                                children=html.Div(
                                    id="attention_weight_similarity_container",
                                    style={
                                        "width": "100%",
                                        "height": "auto",
                                        "overflow": "visible",
                                    },  # Dynamic height and overflow settings
                                ),
                            ),
                            width=6,
                            style={
                                "padding": "100px",
                                "margin-bottom": "10px",
                            },  # Add bottom margin for spacing
                        ),
                    ]
                )
            ]
        )

        layout = html.Div(
            className="main-container center-container",
            style={
                "display": "flex",
                "flex-direction": "column",
                "align-items": "center",
                "justify-content": "flex-start",
            },
            children=[
                select_variant_container,
                training_header,
                training_graph_container,
                view_train_decision,
                filter_layer_section,
                decision_section,
                decision_graph_prompt,
                measure_container,
                clustering_alignment_header,
                clustering_analysis,
                selection_header,
                selection_analysis,
                impact_header,
                training_impact,
            ],
        )
        return layout
