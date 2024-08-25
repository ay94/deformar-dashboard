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
        
        self.plot_button = CustomButton('View Decision Boundary', 'view_decision_boundary').render()
        
    def render(self):
        return SectionContainer(
                    "Decision Boundary Analysis", 
                    [
                    self.decision_type_dropdown,
                    self.decision_columns_dropdown, 
                    self.plot_button  # Include the message
                    ]
                ).render()

       
       

class DistributionSection:
    def __init__(self, config):
        self.distribution_column_dropdown = dcc.Dropdown(
            id='distribution_column',
            multi=False,
            placeholder="Select Distribution column...",
            options=generate_dropdown_options(config.get('statistics_columns', ['Wrong Columns'])),  # Assuming you have a function to generate options
            style={'width': '100%'}  # Assuming you want to use the full width for styling
        )
        self.categorical_column_dropdown = dcc.Dropdown(
            id='categorical_column',
            multi=False,
            placeholder="Select Categorical column...",
            options=generate_dropdown_options(config.get('categorical_columns', ['Wrong Columns'])),  # Assuming you have a function to generate options
            style={'width': '100%'}
        )
        self.plot_button = CustomButton('Plot Distribution', 'plot_distribution').render()
        
    def render(self):
        return SectionContainer(
                    "Distributions", 
                    [
                    self.distribution_column_dropdown,
                    self.categorical_column_dropdown,
                    self.plot_button,  # Include the message
                    ]
                ).render()


class ResultsSection:
    def __init__(self, config):
        self.results_dropdown = dcc.Dropdown(
            id='results_type',
            multi=False,
            placeholder="Select results...",
            options=generate_dropdown_options(config.get('results', []))  # Assuming you have a function to generate options
        )
        self.view_results_type_button = CustomButton('View Results', 'view_results_type').render()
    
    def render(self):
        
        return SectionContainer(
                    "Results", 
                    [
                    self.results_dropdown,
                    self.view_results_type_button,
                    ]
                ).render()


class CustomAnalysisSection:
    def __init__(self, config):
        self.custom_distribution_dropdown = dcc.Dropdown(
            id='custom_analysis_type',
            multi=False,
            placeholder="Select Analysis...",
            options=generate_dropdown_options(config.get('custom_analysis', ['Wrong Columns'])),  # Assuming you have a function to generate options
            style={'width': '100%'}  # Assuming you want to use the full width for styling
        )
        
        self.plot_button = CustomButton('Plot Custom Analysis', 'plot_custom_analysis').render()
        
    def render(self):
        return SectionContainer(
                    "Custom Analysis", 
                    [
                    self.custom_distribution_dropdown,
                    self.plot_button,  # Include the message
                    ]
                ).render()






class DecisionTabLayout:
    def __init__(self, config_manager):
        self.variants = config_manager.variants  # You might want to use config settings if applicable
        self.decision_tab_config = config_manager.decision_tab  # You might want to use config settings if applicable

    def render(self):
        
        select_variant_container = VariantSection(self.variants).render()
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
        
        layout = html.Div(
                    # className='main-container',
                    className='main-container center-container',
                    children=[
                        select_variant_container,
                        decision_container,
                        decision_graph,
                        decision_graph_prompt
                    ],
                    
            )

        return layout




