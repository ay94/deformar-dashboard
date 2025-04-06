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
from dash import html
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
            "TP": "#636EFA",   # soft indigo/periwinkle
            "FP": "#EF553B",   # coral red
            "FN": "#00CC96",   # teal green
            "TN": "#FFB74D",     # soft orange
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
    
def render_model_init():
    return dbc.Container([
        html.Hr(),
        html.H4("Model Initialization", className="text-center", style={"marginBottom": "16px"}),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Button(
                        'Load Models',
                        id='load_model_btn',
                        n_clicks=0,
                        className="btn",
                        style={
                            "backgroundColor": "#3DAFA8",
                            "color": "white",
                            "fontWeight": "bold",
                            "padding": "8px 16px",
                            "borderRadius": "6px",
                            "minWidth": "120px"
                        }
                    ),
                    html.Div(id='model_status', style={
                        "marginTop": "12px",
                        "color": "green",
                        "fontWeight": "bold",
                        "textAlign": "center"
                    })
                ], style={"textAlign": "center"})
            ], width=12),
        ]),

        # html.Hr()
    ], fluid=True)



def render_attention_analysis():
    return dbc.Container([
        # html.Hr(),
        html.H4("Attention Analysis", className="text-center", style={"marginBottom": "16px"}),

        # Dropdowns and Button Row
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='impact_instances',
                    placeholder="Select Sentence id...",
                )
            ], width=3),

            dbc.Col([
                dcc.Dropdown(
                    id='attention_view',
                    placeholder="Select View...",
                    options=[
                        {'label': 'Head View', 'value': 'head'},
                        {'label': 'Model View', 'value': 'model'},
                    ]
                )
            ], width=3),

            dbc.Col([
                CustomButton(
                    "Visualize Training Impact", "visualize_training_impact"
                ).render()
            ], width="auto"),

        ], className="mb-4", justify="center"),

        html.Hr(),

        # Pretrained and Finetuned Visuals
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="loading_bertvis_view_pre",
                    type="default",
                    children=[
                        html.H5("Pretrained", className="text-center"),
                        html.Iframe(id="pre_attention_view", style={"width": "100%", "height": "600px"}),
                    ]
                ),
            ], width=6),
            dbc.Col([
                dcc.Loading(
                    id="loading_bertvis_view_fin",
                    type="default",
                    children=[
                        html.H5("Finetuned", className="text-center"),
                        html.Iframe(id="fin_attention_view", style={"width": "100%", "height": "600px"}),
                    ]
                ),
            ], width=6),
        ], className="mb-4"),


        html.Hr(),

        # Training Impact Heatmap
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="loading_training_impact",
                    type="default",
                    children=[     
                        dcc.Graph(id="instance_training_impact", figure=go.Figure())
                    ]
                )
            ], width="auto")
        ], justify="center"),
    ], fluid=True)

def render_token_analysis():
    return dbc.Container([
        html.Hr(),
        html.H4("Token-Level Analysis", className="text-center", style={"marginBottom": "16px"}),

        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id="core_token_selector",
                    placeholder="Select Token...",
                ),
            ], width=4),
            dbc.Col([
                html.Button(
                    "Compute Token Analysis",
                    id="compute_token_analysis",
                    n_clicks=0,
                    className="btn btn-primary",
                    style={"backgroundColor": "#3DAFA8", "borderColor": "#3DAFA8"}
                )
            ], width="auto"),
        ], justify="center", className="mb-4"),

        html.Hr(),
        html.H5("Prediction Confidence Scores", className="text-center"),
        dcc.Graph(id="token_confidence_scores", figure=go.Figure()),
        html.Hr(),

        
        html.H5("Entity Tag Distribution Across Splits", className="text-center"),
        dcc.Graph(id="token_label_distribution", figure=go.Figure()),
        html.Hr(),


        dbc.Row([
            html.H5("Token Similarity Analysis", className="text-center"),
            dbc.Col([
                html.H5("Train Split", className="text-center"),
                dcc.Loading(
                            id="loading_attention_similarity",
                            type="default",
                            children=[
                                dcc.Graph(id="token_similarity_train"),
                                html.Div(id="token_similarity_table_train", style={"padding": "16px"})
                            ]
                        ),
                # dcc.Graph(id="token_similarity_train"),
                # html.Div(id="token_similarity_table_train", style={"padding": "16px"})
            ], width=6),
            dbc.Col([
                html.H5("Test Split", className="text-center"),
                dcc.Loading(
                            id="loading_attention_similarity",
                            type="default",
                            children=[
                                dcc.Graph(id="token_similarity_test"),
                                html.Div(id="token_similarity_table_test", style={"padding": "16px"})
                            ]
                        ),
            ], width=6),
        ], className="mb-4"),

         

        dbc.Row([
            dbc.Col([
                html.H5("Token Context Viewer", className="text-center"),
                dcc.Dropdown(id="token_view_split_selector", placeholder="Select Split..."),
                dcc.Dropdown(id="token_view_sentence_selector", placeholder="Select Sentence..."),
                html.Div(id="token_sentence_render", style={"padding": "16px"})
            ], width=6)
        ], justify="center"),
        
        html.Hr(),
        
        dbc.Row([
            dbc.Col([
                html.H5("Token Origin Viewer", className="text-center"),
                dcc.Dropdown(id="token_origin_sentence", placeholder="Select Sentence..."),
                html.Div(id="token_origin_sentence_render", style={"padding": "16px"})
            ], width=6)
            
        ], justify="center"),
        
        html.Hr(),
        html.Br(),

    ], fluid=True)

class InstanceTabLayout:
    def __init__(self, config_manager):
        
        self.variants = (
            config_manager.variants
        )  # You might want to use config settings if applicable
        
        self.qualitative_tab_config = (
            config_manager.qualitative
        )  # You might want to use config settings if applicable

    def render(self):

        
        select_variant_container = VariantSection(self.variants).render()
        
        select_instance = html.Div([
            dcc.Dropdown(
                id='instance_selector',
                placeholder="Select an Instance...",
                style={
                    "width": "300px",
                    "margin": "0 auto",  # Center it horizontally
                }
            )
        ], style={"textAlign": "center", "marginTop": "20px", "marginBottom": "30px"})
        
        instance_details = dbc.Container([
            html.Br(),
            select_variant_container,  # ✅ now included
            html.Hr(),
            html.H4("Token-Level Annotations", className="text-center"),
            generate_label_legend(token_color_map),
            html.Br(),
            select_instance,
            render_instance_row("📝 Sentence", "instance_sentence"),
            render_instance_row("✅ Ground Truth", "instance_truth"),
            render_instance_row("🔮 Prediction", "instance_pred"),
            render_instance_row("❌ Correctness", "instance_mistakes", rtl=False),
            html.Div(style={"marginBottom": "30px"}),  # Add spacing here
            
            html.Hr(),
            html.H4("Entity-Level Annotations", className="text-center"),
            generate_label_legend(entity_color_map),
            html.Br(),
            # First: Non-Strict
            html.H5("Scheme: IOB", style={"textAlign": "center", "marginTop": "20px"}),
            render_instance_row("✅ True Entities", "entity_true_iob", rtl=False),
            render_instance_row("🔮 Predicted Entities", "entity_pred_iob", rtl=False),
            render_instance_row("❌ Entity Errors", "entity_error_iob", rtl=False),

            html.Br(),

            # Second: Strict
            html.H5("Scheme: IOB2", style={"textAlign": "center", "marginTop": "20px"}),
            render_instance_row("✅ True Entities", "entity_true_iob2", rtl=False),
            render_instance_row("🔮 Predicted Entities", "entity_pred_iob2", rtl=False),
            render_instance_row("❌ Entity Errors", "entity_error_iob2", rtl=False),
            html.Div(style={"marginBottom": "30px"}),  # Add spacing here
            
            render_model_init(),

            html.Div(style={"marginBottom": "30px"}),  # Add spacing here
            
            render_token_analysis(),  # ⬅️ Insert this
            render_attention_analysis(),

        ], fluid=True, style={"marginTop": "20px"})
        
        
        
        layout = html.Div(
            className="main-container center-container",
            style={
                "display": "flex",
                "flex-direction": "column",
                "align-items": "center",
                "justify-content": "flex-start",
            },
            children=[
                # select_variant_container,
                instance_details
            
            ],
        )
        return layout
