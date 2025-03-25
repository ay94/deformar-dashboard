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


class InstanceTabLayout:
    def __init__(self, config_manager):
        
        self.variants = (
            config_manager.variants
        )  # You might want to use config settings if applicable
        
        self.qualitative_tab_config = (
            config_manager.qualitative
        )  # You might want to use config settings if applicable

    def render(self):

        
        layout = dbc.Container([
            html.H3("Instance-Level Model Analysis", className="my-3"),

            dbc.Row([
                # === Left Sidebar (Controls) ===
                dbc.Col([
                    html.Label("Select Variant"),
                    dcc.Dropdown(
                        id="variant_selector",
                        options=[
                            {"label": "ANERCorp_CamelLab_arabertv02", "value": "ANERCorp_CamelLab_arabertv02"},
                            {"label": "conll2003_bert", "value": "conll2003_bert"},
                        ],
                        placeholder="Choose a variant",
                    ),
                    html.Br(),

                    html.Label("Select Instance ID"),
                    dcc.Dropdown(id="instance_selector", placeholder="Choose sentence..."),
                    html.Br(),

                    html.Button("Run Inference", id="run_button", className="btn btn-primary"),
                ], width=4),

                # === Main Display ===
                dbc.Col([
                    html.Div(id="output_display", className="p-3 border"),
                ], width=8)
            ])
        ], fluid=True)
        return layout
