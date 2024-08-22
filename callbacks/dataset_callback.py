from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from dash import  dcc, html
import plotly.graph_objs as go

from utils.tab_managers import DatasetTabManager

def register_callbacks(app, variants_data):
    tab_manager = DatasetTabManager(variants_data)

    @app.callback(
        Output('loaded_tab_data', 'children'),
        Input('variant_selector', 'value')
    )
    def update_output(variant):
        tab_data = tab_manager.get_tab_data(variant)
        if tab_data:
            return html.Div(
                        f"Tab data loaded and Variant {variant} is chosen.",
                        className='prompt-message'
                    )
        else:
            return html.Div(
                        "Tab data is not available",
                        className='prompt-message'
                    )
    @app.callback(
        Output("summary_container", "children"),
        [
            Input('generate_summary', 'n_clicks'),
            State('statistical_columns', 'value'), 
            State('variant_selector', 'value')
        ]
    )
    def generate_summary_statistics(n_clicks, statistical_columns, variant):
        if n_clicks == 0:
            raise PreventUpdate

        # Generate the summary statistics
        summary_table = tab_manager.generate_summary_statistics(variant, statistical_columns)
        
        if summary_table is None:
            return html.Div(
                "Please select a type of results and click 'View Results' to view data.",
                className='prompt-message'
            )
        return summary_table 
    
   

    
    @app.callback(
        Output("distribution_container", "children"),
        [
            Input('plot_distribution', 'n_clicks'),
            State('distribution_column', 'value'),
            State('categorical_column', 'value'),
            State('variant_selector', 'value')
        ]
    )
    def plot_distribution(n_clicks, distribution_column, categorical_column, variant):
        if n_clicks == 0:
            raise PreventUpdate

        # Generate the figure using the TabManager's method
        figure = tab_manager.generate_distribution_or_violin(
            variant=variant,
            distribution_column=distribution_column,
            categorical_column=categorical_column,
        )

        # Check if a valid figure was returned
        if isinstance(figure, go.Figure):
            return dcc.Graph(figure=figure)
        
        return html.Div(
                "Please select a  type and click 'Plot Distribution' to view plot'.",
                className='prompt-message'
            )

    
    @app.callback(
        Output("results_output_container", "children"),
        [
            Input('view_results_type', 'n_clicks'),
            State('results_type', 'value'),
            State('variant_selector', 'value')
        ]
    )
    def view_results(n_clicks, results_type, variant):
        if n_clicks == 0:
            raise PreventUpdate
        
        # Get the results data using the tab manager
        results_data = tab_manager.get_results_data(variant, results_type)
        
        if results_data is None:
            return html.Div(
                "Please select a type of results and click 'View Results' to view data.",
                className='prompt-message'
            )
        
        # Use the CustomDataTable class to create the table
        return results_data

    
    @app.callback(
        Output("custom_distribution_graph_container", "children"),
        [Input('plot_custom_distribution', 'n_clicks')],
        [State('custom_distribution_type', 'value'), State('variant_selector', 'value')]
    )
    def custom_distributions_callback(n_clicks, custom_distribution_type, variant):
        if n_clicks == 0:
            raise PreventUpdate

        result = tab_manager.custom_distributions(custom_distribution_type, variant)

        if result is None:
            return html.Div(
                "Please select a  type and click 'Plot Custom Distribution' to view data'.",
                className='prompt-message'
            )

        return result

    