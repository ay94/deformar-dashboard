from dash import html, dcc
from dash import html, dash_table
import plotly.graph_objs as go




def get_input_trigger(ctx):
    return ctx.triggered[0]["prop_id"].split(".")[0]

def generate_dropdown_options(columns):
    # This could pull column names from a dataset configuration or similar
    return [{'label': col, 'value': col} for col in columns]


def generate_status_table(variants_data):
    """Generates a Dash DataTable showing the loading status for attributes across multiple variants."""
    if not variants_data:
        return html.Div("No data loaded.")

    data = []
    attributes = set()
    for variant, dashboard_data in variants_data.items():
        row = {'Variant': variant}
        for attr in dashboard_data.__dict__.keys():
            status = "Loaded" if dashboard_data.is_loaded(attr) else "Empty"
            row[attr] = status
            attributes.add(attr)
        data.append(row)

    columns = [{"name": "Variant", "id": "Variant"}] + [{"name": attr, "id": attr} for attr in sorted(attributes)]
    
    return dash_table.DataTable(
        data=data,
        columns=columns,
        style_data_conditional=[
            {
                'if': {
                    'filter_query': f'{{{attr}}} = "Loaded"',
                    'column_id': attr
                },
                'color': 'green',
                'fontWeight': 'bold'
            }
            for attr in attributes if attr != 'Variant'  # Avoiding styling the 'Variant' column
        ] + [
            {
                'if': {
                    'filter_query': f'{{{attr}}} = "Empty"',
                    'column_id': attr
                },
                'color': 'red',
                'fontWeight': 'bold'
            }
            for attr in attributes if attr != 'Variant'
        ],
        
        style_cell={'textAlign': 'center', 'padding': '10px'},
        style_header={
            'text-align': 'center', 
            'background-color': '#3DAFA8',
            'color': 'white'
        },
        style_as_list_view=True
    )



class CustomButton:
    def __init__(self, label, button_id, class_name='button-common', **kwargs):
        self.label = label
        self.id = button_id
        self.class_name = class_name
        self.kwargs = kwargs

    def render(self):
        return html.Button(self.label, id=self.id, className=self.class_name, **self.kwargs)


class CustomDataTable:
    def __init__(self, table_id, data=None, columns=None, style_header=None, page_size=10, **kwargs):
        if table_id is None:
            raise ValueError("A valid table_id must be provided.")
        self.table_id = table_id
        self.data = data if data is not None else []  # Default to an empty list if no data is provided
        self.columns = columns if columns is not None else []  # Default to an empty list if no columns are provided
        self.style_header = style_header or {'text-align': 'center', 'background-color': '#3DAFA8', 'color': 'white'}
        self.page_size = page_size
        self.kwargs = kwargs

    def render(self):
        return dash_table.DataTable(
            id=self.table_id,
            columns=self.columns,
            data=self.data,
            style_header=self.style_header,
            editable=True,
            filter_action="native",
            sort_action="native",
            sort_mode='multi',
            column_selectable="single",
            row_selectable='multi',
            row_deletable=True,
            page_action='native',
            page_current=0,
            page_size=self.page_size,
            **self.kwargs
        )

class DropdownContainer:
        def __init__(self, label_text, dropdown, output_container):
            self.label_text = label_text
            self.dropdown = dropdown
            self.output_container = output_container
            
        def render(self):
            return html.Div(
                    className='dropdown-container',
                    children=[
                        html.Div(
                            className='inner',
                            children=[
                                html.Label(self.label_text, className='custom-label'),
                                self.dropdown,
                                self.output_container
                            ], 
                        )
                    ],
                )

class SectionContainer:
    def __init__(self, header_text, content_components):
        self.header_text = header_text
        self.content_components = content_components
    
    def render(self):
        return html.Div(
                className='section-container',
                children=[
                    html.H2(self.header_text, className='section-header',),
                    html.Div(
                        className='inner',
                        children=self.content_components,
    
                    )
                ],
            )