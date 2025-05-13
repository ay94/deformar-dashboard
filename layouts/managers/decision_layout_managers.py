import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import dcc, html

from layouts.managers.layout_managers import (CustomButton, LoadingContainer,
                                              SectionContainer, VariantSection,
                                              FilterLayerSection,
                                              generate_dropdown_options, generate_variant_dropdown_options)
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
            multi=True,
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
            style_table={"overflowX": "auto"},
            style_cell={"minWidth": "120px", "width": "120px", "maxWidth": "200px", "whiteSpace": "normal"},
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
                        html.Div(
                            self.data_table,
                            style={
                                "overflowX": "auto",
                                "width": "100%",
                            },
                        )
                    ],
                ),
                self.filtered,
            ],
            style={
                # "width": "95%",
                # "display": "flex",
                # "justifyContent": "center",
                 "width": "100%",
                "maxWidth": "100%",
                "overflowX": "auto",  # Make outer container scrollable too
            },
        )

        return FilterLayerSection(
            'Filtering Section',
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
            placeholder="Colour By (UMAP Representation View)...",
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
            placeholder="Colour By (Behavioural Metric View)...",
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
            placeholder="Select Numerical Variables...",
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
            "Behavioural Analysis",
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

token_color_map = {
            "O": "saddlebrown",
            "B-PER": "deepskyblue",
            "I-PER": "lightblue",
            "B-ORG": "darkcyan",
            "I-ORG": "cyan",
            "B-LOC": "darkgreen",
            "I-LOC": "yellowgreen",
            "B-MISC": "palevioletred",
            "I-MISC": "violet",
            "IGNORED": "grey",
            "[CLS]": "grey",
            "[SEP]": "grey",
        }

entity_color_map = {
            "LOC": "darkgreen",
            "PER": "deepskyblue",
            "ORG": "darkcyan",
            "MISC": "palevioletred",
            "FP": "#EF553B",   # coral red
            "FN": "#00CC96",   # teal green
            "No Errors": "green"
        }


def generate_label_legend(color_map: dict):
    return html.Div([
        html.H6("Label Color Map", style={"marginBottom": "10px", "fontWeight": "bold"}),
        html.Div([
            html.Div(label, style={
                "backgroundColor": color,
                "color": "white",
                "padding": "4px 8px",
                "borderRadius": "6px",
                "margin": "4px",
                "display": "inline-block",
                "fontSize": "0.85rem",
                "fontWeight": "bold",
                "minWidth": "60px",
                "textAlign": "center",
            }) for label, color in color_map.items()
        ], style={
            "display": "flex",
            "flexWrap": "wrap",
            "justifyContent": "center",  # 👈 center
            "gap": "4px",
            "padding": "10px",
            "border": "1px solid #ddd",
            "borderRadius": "8px",
            "backgroundColor": "#f9f9f9",
            "maxWidth": "90%",
        })
    ])

def render_instance_row(label, content_id, rtl=True):
    return html.Div([
        html.H6(f"{label}:", style={"minWidth": "140px", "marginBottom": "0px"}),
        html.Div(id=content_id, style={
            "padding": "10px",
            "marginRight": "10px",
            "direction": "rtl" if rtl else "ltr",
            "unicodeBidi": "embed"
        }),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"})
    
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
        select_instance = html.Div([
            dcc.Dropdown(
                id='training_sentences',
                placeholder="Select an Sentence...",
                style={
                    "width": "300px",
                    "margin": "0 auto",  # Center it horizontally
                }
            )
        ], style={"textAlign": "center", "marginTop": "20px", "marginBottom": "30px"})
        instance_details = dbc.Container([
            html.Br(),
            html.Hr(),
            html.H4("Token-Level Annotations", className="text-center"),
            generate_label_legend(token_color_map),
            html.Br(),
            select_instance,
            render_instance_row("📝 Sentence", "training_sentence"),
            render_instance_row("✅ Ground Truth", "training_truth"),
            html.Div(style={"marginBottom": "30px"}),  # Add spacing here
            
            html.Hr(),
            html.H4("Entity-Level Annotations", className="text-center"),
            generate_label_legend(entity_color_map),
            html.Br(),
            # First: Non-Strict
            html.H5("Scheme: IOB", style={"textAlign": "center", "marginTop": "20px"}),
            render_instance_row("✅ True Entities", "training_entity_true_iob", rtl=False),

            html.Br(),

            # Second: Strict
            html.H5("Scheme: IOB2", style={"textAlign": "center", "marginTop": "20px"}),
            render_instance_row("✅ True Entities", "training_entity_true_iob2", rtl=False),
            html.Div(style={"marginBottom": "30px"}),  # Add spacing here
            

        ], fluid=True, style={"marginTop": "20px"})
        
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
                instance_details,
                filter_layer_section,
                decision_section,
                decision_graph_prompt,
                measure_container,
                selection_header,
                selection_analysis,
                clustering_alignment_header,
                clustering_analysis,
                impact_header,
                training_impact,
            ],
        )
        return layout
