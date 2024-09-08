import json
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dash_table, dcc, html


def get_input_trigger(ctx):
    return ctx.triggered[0]["prop_id"].split(".")[0]


def process_json_data(jsonData):
    if not jsonData:
        return pd.DataFrame()
    if isinstance(jsonData, str):
        data = json.loads(jsonData)
    else:
        data = jsonData
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        df = pd.read_json(StringIO(json.dumps(data)), orient="split")
    else:
        return []
    return df


def process_selection(decisionSelection):
    selection_ids = []
    if decisionSelection:
        if isinstance(decisionSelection, str) and decisionSelection.strip():
            decision_selection = process_json_data(decisionSelection)
            if not decision_selection.empty:
                selection_ids = decision_selection["Global Id"].tolist()
        else:
            decision_selection = pd.DataFrame()
    return selection_ids


def generate_dropdown_options(columns):
    # This could pull column names from a dataset configuration or similar
    return [{"label": col, "value": col} for col in columns]


def generate_status_table(variants_data):
    """Generates a Dash DataTable showing the loading status for attributes across multiple variants."""
    if not variants_data:
        return html.Div("No data loaded.")

    data = []
    attributes = set()
    for variant, dashboard_data in variants_data.items():
        row = {"Variant": variant}
        for attr in dashboard_data.__dict__.keys():
            status = "Loaded" if dashboard_data.is_loaded(attr) else "Empty"
            row[attr] = status
            attributes.add(attr)
        data.append(row)

    columns = [{"name": "Variant", "id": "Variant"}] + [
        {"name": attr, "id": attr} for attr in sorted(attributes)
    ]

    return dash_table.DataTable(
        data=data,
        columns=columns,
        style_data_conditional=[
            {
                "if": {"filter_query": f'{{{attr}}} = "Loaded"', "column_id": attr},
                "color": "green",
                "fontWeight": "bold",
            }
            for attr in attributes
            if attr != "Variant"  # Avoiding styling the 'Variant' column
        ]
        + [
            {
                "if": {"filter_query": f'{{{attr}}} = "Empty"', "column_id": attr},
                "color": "red",
                "fontWeight": "bold",
            }
            for attr in attributes
            if attr != "Variant"
        ],
        style_cell={"textAlign": "center", "padding": "10px"},
        style_header={
            "text-align": "center",
            "background-color": "#3DAFA8",
            "color": "white",
        },
        style_as_list_view=True,
    )


class CustomButton:
    def __init__(self, label, button_id, class_name="button-common", **kwargs):
        self.label = label
        self.id = button_id
        self.class_name = class_name
        self.kwargs = kwargs

    def render(self):
        return html.Button(
            self.label, id=self.id, className=self.class_name, **self.kwargs
        )


# class CustomDataTable:
#     def __init__(self, table_id, data=None, columns=None, style_header=None, page_size=10, **kwargs):
#         if table_id is None:
#             raise ValueError("A valid table_id must be provided.")
#         self.table_id = table_id
#         self.data = data if data is not None else []  # Default to an empty list if no data is provided
#         self.columns = columns if columns is not None else []  # Default to an empty list if no columns are provided
#         self.style_header = style_header or {'text-align': 'center', 'background-color': '#3DAFA8', 'color': 'white'}
#         self.page_size = page_size
#         self.kwargs = kwargs

#     def render(self):
#         return dash_table.DataTable(
#             id=self.table_id,
#             columns=self.columns,
#             data=self.data,
#             style_header=self.style_header,
#             editable=True,
#             filter_action="native",
#             sort_action="native",
#             sort_mode='multi',
#             column_selectable="single",
#             row_selectable='multi',
#             row_deletable=True,
#             page_action='native',
#             page_current=0,
#             page_size=self.page_size,
#             **self.kwargs
#         )


class CustomDataTable:
    def __init__(
        self,
        table_id,
        data=None,
        columns=None,
        style_header=None,
        page_size=10,
        **kwargs,
    ):
        if table_id is None:
            raise ValueError("A valid table_id must be provided.")
        self.table_id = table_id
        self.data = (
            data if data is not None else []
        )  # Default to an empty list if no data is provided
        self.columns = (
            columns if columns is not None else []
        )  # Default to an empty list if no columns are provided
        self.style_header = style_header or {
            "text-align": "center",
            "background-color": "#3DAFA8",
            "color": "white",
        }
        self.page_size = page_size
        self.kwargs = kwargs

    def render(self):
        return dash_table.DataTable(
            id=self.table_id,
            columns=self.columns,
            data=self.data,
            style_header=self.style_header,
            style_table={
                "overflowX": "auto",  # Enable horizontal scrolling
                "minWidth": "100%",  # Ensures the table takes at least the full width of its container
            },
            style_cell={
                "minWidth": "150px",
                "width": "150px",
                "maxWidth": "300px",  # Adjust these values as needed
                "whiteSpace": "normal",  # Wrap text within cells
                "overflow": "hidden",  # Hide overflow text
                "textOverflow": "ellipsis",  # Show ellipsis for overflow text
            },
            editable=True,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            column_selectable="single",
            row_selectable="multi",
            row_deletable=True,
            selected_rows=[],
            page_action="native",
            page_current=0,
            page_size=self.page_size,
            **self.kwargs,
        )


class DropdownContainer:
    def __init__(self, label_text, dropdown, output_container):
        self.label_text = label_text
        self.dropdown = dropdown
        self.output_container = output_container

    def render(self):
        return html.Div(
            className="dropdown-container",
            children=[
                html.Div(
                    className="inner",
                    children=[
                        html.Label(self.label_text, className="custom-label"),
                        self.dropdown,
                        self.output_container,
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
            className="section-container",
            children=[
                html.H2(
                    self.header_text,
                    className="section-header",
                ),
                html.Div(
                    className="inner",
                    children=self.content_components,
                ),
            ],
        )


# class LoadingContainer:
#     def __init__(self, container_id, loader_id, container_style=None, inner_container_style=None, loader_style=None):
#         """
#         Initialize the LoadingContainer with the necessary components.

#         :param container_id: The ID for the inner container that will be updated dynamically.
#         :param loader_id: The ID for the dcc.Loading component.
#         :param container_style: Optional dictionary of CSS styles for the container.
#         :param loader_style: Optional dictionary of CSS styles for the loader.
#         """
#         # Set default styles if none are provided
#         if container_style is None:
#             self.container_style = {'width': '100%', 'height': '500px'}
#         self.container_style = container_style
#         if inner_container_style is None:
#             inner_container_style = {'width': '100%', 'height': '600px'}
#         if loader_style is None:
#             loader_style = {'width': '100%', 'height': '100%'}

#         # Define the inner container where content will be dynamically loaded
#         self.inner_container = html.Div(
#             id=container_id,
#             style=inner_container_style
#         )

#         # Define the loading component with a spinner
#         self.loading_component = dcc.Loading(
#             id=loader_id,
#             type="default",
#             children=[self.inner_container],
#             style=loader_style
#         )

#     def render(self):
#         """
#         Render the loading container with its components.

#         :return: The Dash component structure for this loading container.
#         """
#         return html.Div(
#             children=[self.loading_component],
#             style=  self.container_style# Ensure the outer container has a fixed or flexible size
#         )


class LoadingContainer:
    def __init__(
        self,
        container_id,
        loader_id,
        container_style=None,
        inner_container_style=None,
        loader_style=None,
    ):
        """
        Initialize the LoadingContainer with the necessary components.

        :param container_id: The ID for the inner container that will be updated dynamically.
        :param loader_id: The ID for the dcc.Loading component.
        :param container_style: Optional dictionary of CSS styles for the outer container.
        :param inner_container_style: Optional dictionary of CSS styles for the inner container.
        :param loader_style: Optional dictionary of CSS styles for the loader.
        """
        # Set default styles if none are provided
        if container_style is None:
            container_style = {"width": "100%", "height": "500px"}
        if inner_container_style is None:
            inner_container_style = {
                "width": "100%",
                "height": "100%",  # Make the inner container fill the entire outer container
            }
        if loader_style is None:
            loader_style = {
                "width": "100%",
                "height": "100%",  # Make the loader fill the inner container
            }

        # Define the inner container where content will be dynamically loaded
        self.inner_container = html.Div(id=container_id, style=inner_container_style)

        # Define the loading component with a spinner
        self.loading_component = dcc.Loading(
            id=loader_id,
            type="default",
            children=[self.inner_container],
            style=loader_style,
        )

        # Save the outer container style
        self.container_style = container_style

    def render(self):
        """
        Render the loading container with its components.

        :return: The Dash component structure for this loading container.
        """
        return html.Div(
            children=[self.loading_component],
            style=self.container_style,  # Apply the consistent container style
        )


class VariantSection:
    def __init__(self, variants):
        self.variants_dropdown = dcc.Dropdown(
            id="variant_selector",
            placeholder="Select variant...",
            options=generate_dropdown_options(variants),
            value=variants[0],
            style={"width": "50%", "margin": "auto"},
        )
        self.tab_data_div = html.Div(id="loaded_tab_data")

    def render(self):

        return DropdownContainer(
            "Select Variant:", self.variants_dropdown, self.tab_data_div
        ).render()
