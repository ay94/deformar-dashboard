from dash import html, dcc
from dash import html
from utils.layout_managers import CustomButton, generate_dropdown_options, DropdownContainer, SectionContainer





       
class VariantSection:
    def __init__(self, variants):
        self.variants_dropdown = dcc.Dropdown(
            id='variant_selector',
            placeholder="Select column...",
            options=generate_dropdown_options(variants),
            value=variants[0],
            style={'width': '50%', 'margin': 'auto'}
        )
        self.tab_data_div = html.Div(id='loaded_tab_data')

    def render(self):
        
        return DropdownContainer(
            "Select Variant:", 
            self.variants_dropdown,
            self.tab_data_div
        ).render()




class SummarySection:
    def __init__(self, config):
        self.statistical_dropdown = dcc.Dropdown(
            id='statistical_columns',
            multi=True,
            placeholder="Select column...",
            options=generate_dropdown_options(config.get('statistics_columns'))  # Assuming you have a function to generate options
        )
        self.generate_summary_button = CustomButton('Generate Summary Statistics', 'generate_summary').render()
    

    def render(self):
        
        return SectionContainer(
                    "Summary Statistics", 
                    [
                    self.statistical_dropdown,
                    self.generate_summary_button,
                    ]
                ).render()
       
       

class DistributionSection:
    def __init__(self, config):
        self.distribution_column_dropdown = dcc.Dropdown(
            id='distribution_column',
            multi=False,
            placeholder="Select Distribution column...",
            options=generate_dropdown_options(config.get('statistics_columns')),  # Assuming you have a function to generate options
            style={'width': '100%'}  # Assuming you want to use the full width for styling
        )
        self.categorical_column_dropdown = dcc.Dropdown(
            id='categorical_column',
            multi=False,
            placeholder="Select Categorical column...",
            options=generate_dropdown_options(config.get('categorical_columns')),  # Assuming you have a function to generate options
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
            options=generate_dropdown_options(config.get('results'))  # Assuming you have a function to generate options
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



class CustomDistributionSection:
    def __init__(self, config):
        self.custom_distribution_dropdown = dcc.Dropdown(
            id='custom_distribution_type',
            multi=False,
            placeholder="Select Distribution...",
            options=generate_dropdown_options(config.get('custom_distributions')),  # Assuming you have a function to generate options
            style={'width': '100%'}  # Assuming you want to use the full width for styling
        )
        
        self.plot_button = CustomButton('Plot Custom Distribution', 'plot_custom_distribution').render()
        
    def render(self):
        return SectionContainer(
                    "Custom Distributions", 
                    [
                    self.custom_distribution_dropdown,
                    self.plot_button,  # Include the message
                    ]
                ).render()



class DatasetTabLayout:
    def __init__(self, config_manager):
        self.variants = config_manager.variants  # You might want to use config settings if applicable
        self.dataset_tab_config = config_manager.dataset_tab  # You might want to use config settings if applicable

    def render(self):
        
        select_variant_container = VariantSection(self.variants).render()
        #  Summary Statistics Section Container 
        summary_container = SummarySection(self.dataset_tab_config).render()  # Create and render the distribution section
        summary_statistics_container = html.Div(
            id="summary_container", 
            style={'width': '50%', 'height': '100%'}  # Initially hidden
        )
        
        # Distribution Section Container 
        distribution_container = DistributionSection(self.dataset_tab_config).render()  # Create and render the distribution section
        
        distribution_graph_container = html.Div(
            id="distribution_container", 
            style={'width': '100%', 'height': '100%'}  # Initially hidden
        )
        
        results_container = ResultsSection(self.dataset_tab_config).render()  
        
        results_data_table = html.Div(
            id="results_output_container", 
            style={'width': '50%', 'height': '100%'}  # Initially hidden
        )
        
        custom_distribution_container = CustomDistributionSection(self.dataset_tab_config).render()
        
        custom_distribution_graph_container = html.Div(
            id="custom_distribution_graph_container", 
            style={'width': '100%', 'height': '100%'}  # Initially hidden
        )
        
        layout = html.Div(
                    className='main-container',
                    children=[
                        select_variant_container,
                        summary_container,
                        summary_statistics_container,
                        distribution_container,
                        distribution_graph_container,
                        results_container,
                        results_data_table,
                        custom_distribution_container,
                        custom_distribution_graph_container
                    ]
            )

        return layout




