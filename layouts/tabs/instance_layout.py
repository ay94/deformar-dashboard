# from . import dash_table, dcc, go, html


def get_layout():
    pass


#     return dcc.Tab(
#         label="Instance Level Analysis",
#         value="instance",
#         children=[
#             html.H1("Instance Analysis", style={"text-align": "center"}),
#             html.Div(
#                 style={
#                     "display": "flex",
#                     "justify-content": "center",
#                     "flex-direction": "column",
#                     "align-items": "center",
#                     "height": "50vh",
#                 },
#                 children=[
#                     dcc.Dropdown(
#                         id="error_instances",
#                         placeholder="Select Sentence id...",
#                         style={
#                             "width": "300px",
#                             "margin": "4px 2px",
#                         },
#                     ),
#                     html.Button(
#                         "Visualize Instance",
#                         id="visualize_instance",
#                         n_clicks=0,
#                         style={
#                             "background-color": "#3DAFA8",
#                             "color": "white",
#                             "border": "none",
#                             "padding": "10px 20px",
#                             "text-align": "center",
#                             "text-decoration": "none",
#                             "display": "inline-block",
#                             "font-size": "16px",
#                             "margin": "4px 2px",
#                             "cursor": "pointer",
#                             "border-radius": "4px",
#                         },
#                     ),
#                     html.Div(id="initialize_instance_tab"),
#                 ],
#             ),
#             html.Div(
#                 id="instance-container",
#                 children=[
#                     html.Div(
#                         [
#                             html.H3("Label Color Map:"),
#                             html.Div(id="instance_label_map"),
#                         ],
#                         style={"display": "flex", "align-items": "center"},
#                     ),
#                     html.Div(
#                         [
#                             html.H3("Instance Sentence:"),
#                             html.Div(
#                                 id="instance_sentence",
#                                 style={
#                                     "padding": "10px",
#                                     "margin-right": "10px",
#                                     "direction": "rtl",
#                                     "unicode-bidi": "embed",
#                                 },
#                             ),
#                         ],
#                         style={"display": "flex", "align-items": "center"},
#                     ),
#                     html.Div(
#                         [
#                             html.H3("Instance Truth:"),
#                             html.Div(
#                                 id="instance_truth",
#                                 style={
#                                     "padding": "10px",
#                                     "margin-right": "10px",
#                                     "direction": "rtl",
#                                     "unicode-bidi": "embed",
#                                 },
#                             ),
#                         ],
#                         style={"display": "flex", "align-items": "center"},
#                     ),
#                     html.Div(
#                         [
#                             html.H3("Instance Preds:"),
#                             html.Div(
#                                 id="instance_pred",
#                                 style={
#                                     "padding": "10px",
#                                     "margin-right": "10px",
#                                     "direction": "rtl",
#                                     "unicode-bidi": "embed",
#                                 },
#                             ),
#                         ],
#                         style={"display": "flex", "align-items": "center"},
#                     ),
#                     html.Div(
#                         [
#                             html.H3("Instance Mistakes:"),
#                             html.Div(
#                                 id="instance_mistakes",
#                                 style={
#                                     "padding": "10px",
#                                     "margin-right": "10px",
#                                 },
#                             ),
#                         ],
#                         style={"display": "flex", "align-items": "center"},
#                     ),
#                 ],
#             ),
#             html.H3("Instance Scatter Plot", style={"text-align": "center"}),
#             html.Div(
#                 children=[
#                     dcc.Loading(
#                         id="instance_loading",
#                         type="default",
#                         children=[
#                             dcc.Graph(id="instance_scatter", figure=go.Figure()),
#                         ],
#                     ),
#                 ],
#                 style={"margin-bottom": "20px"},
#             ),
#             html.H3(
#                 "Example Attention / Activation Plots", style={"text-align": "center"}
#             ),
#             html.Div(
#                 style={
#                     "display": "flex",
#                     "justify-content": "center",
#                     "align-items": "center",
#                     "height": "50vh",
#                 },
#                 children=[
#                     dcc.Dropdown(
#                         id="impact_instances",
#                         placeholder="Select Sentence id...",
#                         style={
#                             "width": "200px",
#                             "margin": "4px 2px",
#                         },
#                     ),
#                     dcc.Dropdown(
#                         id="attention_view",
#                         placeholder="Select View...",
#                         options=[
#                             {"label": "Head View", "value": "head"},
#                             {"label": "Model View", "value": "model"},
#                         ],
#                         style={"width": "200px", "margin": "4px 2px"},
#                     ),
#                     html.Button(
#                         "Visualize Training Impact",
#                         id="visualize_training_impact",
#                         n_clicks=0,
#                         style={
#                             "background-color": "#3DAFA8",
#                             "color": "white",
#                             "border": "none",
#                             "padding": "10px 20px",
#                             "text-align": "center",
#                             "text-decoration": "none",
#                             "display": "inline-block",
#                             "font-size": "16px",
#                             "margin": "4px 2px",
#                             "cursor": "pointer",
#                             "border-radius": "4px",
#                         },
#                     ),
#                     html.Div(id="initialize_instance"),
#                 ],
#             ),
#             html.Div(
#                 children=[
#                     html.H3("Pretrained"),
#                     html.Iframe(
#                         id="pre_attention_view",
#                         style={"width": "60%", "height": "1000px"},
#                     ),
#                     html.H3("Finetuned"),
#                     html.Iframe(
#                         id="fin_attention_view",
#                         style={"width": "60%", "height": "1000px"},
#                     ),
#                     dcc.Graph(id="instance_training_impact", figure=go.Figure()),
#                 ],
#                 style={
#                     "display": "flex",
#                     "justify-content": "center",
#                     "align-items": "center",
#                 },
#             ),
#             html.Div(
#                 style={
#                     "display": "flex",
#                     "justify-content": "center",
#                     "align-items": "center",
#                     "height": "50vh",
#                 },
#                 children=[
#                     html.H1("Token Analysis"),
#                 ],
#             ),
#             html.Div(
#                 style={
#                     "display": "flex",
#                     "justify-content": "center",
#                     "align-items": "center",
#                     "height": "50vh",
#                 },
#                 children=[
#                     dcc.Dropdown(
#                         id="instance_tokens",
#                         placeholder="Select Token...",
#                         style={
#                             "width": "300px",
#                             "margin": "4px 2px",
#                         },
#                     ),
#                     dcc.Dropdown(
#                         id="example_split",
#                         placeholder="Select Split...",
#                         options=[
#                             {"label": "Train", "value": "train"},
#                             {"label": "Test", "value": "test"},
#                         ],
#                         style={"width": "200px", "margin": "4px 2px"},
#                     ),
#                     html.Button(
#                         "Load Token Data",
#                         id="load_token_data",
#                         n_clicks=0,
#                         style={
#                             "background-color": "#3DAFA8",
#                             "color": "white",
#                             "border": "none",
#                             "padding": "10px 20px",
#                             "text-align": "center",
#                             "text-decoration": "none",
#                             "display": "inline-block",
#                             "font-size": "16px",
#                             "margin": "4px 2px",
#                             "cursor": "pointer",
#                             "border-radius": "4px",
#                         },
#                     ),
#                 ],
#             ),
#             html.Div(
#                 [
#                     dcc.Dropdown(
#                         id="choose_example",
#                         placeholder="Select Instance...",
#                         style={"width": "300px"},
#                     ),
#                     html.Button(
#                         "Visualize Example",
#                         id="visualize_token_example",
#                         n_clicks=0,
#                         style={
#                             "background-color": "#3DAFA8",
#                             "color": "white",
#                             "border": "none",
#                             "padding": "10px 20px",
#                             "text-align": "center",
#                             "text-decoration": "none",
#                             "display": "inline-block",
#                             "font-size": "16px",
#                             "margin": "4px 2px",
#                             "cursor": "pointer",
#                             "border-radius": "4px",
#                         },
#                     ),
#                     html.Div(id="examples_status"),
#                 ],
#                 style={
#                     "width": "30%",
#                     "display": "inline-block",
#                 },
#             ),
#             html.Div(
#                 id="example-container",
#                 children=[
#                     html.Div(
#                         [
#                             html.H3("Label Color Map:"),
#                             html.Div(id="example_label_map"),
#                         ],
#                         style={"display": "flex", "align-items": "center"},
#                     ),
#                     html.Div(
#                         [
#                             html.H3("Example Sentence:"),
#                             html.Div(
#                                 id="example_sentence",
#                                 style={
#                                     "padding": "10px",
#                                     "margin-right": "10px",
#                                     "direction": "rtl",
#                                     "unicode-bidi": "embed",
#                                 },
#                             ),
#                         ],
#                         style={"display": "flex", "align-items": "center"},
#                     ),
#                     html.Div(
#                         [
#                             html.H3("Example Truth:"),
#                             html.Div(
#                                 id="example_truth",
#                                 style={
#                                     "padding": "10px",
#                                     "margin-right": "10px",
#                                     "direction": "rtl",
#                                     "unicode-bidi": "embed",
#                                 },
#                             ),
#                         ],
#                         style={"display": "flex", "align-items": "center"},
#                     ),
#                     html.Div(
#                         [
#                             html.H3("Example Preds:"),
#                             html.Div(
#                                 id="example_pred",
#                                 style={
#                                     "padding": "10px",
#                                     "margin-right": "10px",
#                                     "direction": "rtl",
#                                     "unicode-bidi": "embed",
#                                 },
#                             ),
#                         ],
#                         style={"display": "flex", "align-items": "center"},
#                     ),
#                     html.Div(
#                         [
#                             html.H3("Example Mistakes:"),
#                             html.Div(
#                                 id="example_mistakes",
#                                 style={
#                                     "padding": "10px",
#                                     "margin-right": "10px",
#                                 },
#                             ),
#                         ],
#                         style={"display": "flex", "align-items": "center"},
#                     ),
#                 ],
#             ),
#             html.Div(
#                 children=[
#                     html.H3("Example Scatter Plot", style={"text-align": "center"}),
#                     dcc.Loading(
#                         id="example_loading",
#                         type="default",
#                         children=[
#                             dcc.Graph(id="example_scatter", figure=go.Figure()),
#                         ],
#                     ),
#                     html.H3("Examples Scatter Plot", style={"text-align": "center"}),
#                     dcc.Loading(
#                         id="examples_loading",
#                         type="default",
#                         children=[
#                             dcc.Graph(id="examples_scatter", figure=go.Figure()),
#                         ],
#                     ),
#                 ],
#                 style={"margin-bottom": "20px"},
#             ),
#             html.H3("Example Token Similarity Matrix", style={"text-align": "center"}),
#             html.Div(
#                 children=[
#                     dcc.Dropdown(
#                         id="example_tokens_similarity",
#                         multi=True,
#                         placeholder="Select Tokens...",
#                         style={"width": "300px"},
#                     ),
#                     html.Button(
#                         "Compute Example Similarity Matrix",
#                         id="compute_example_similarity_matrix",
#                         n_clicks=0,
#                         style={
#                             "background-color": "#3DAFA8",
#                             "color": "white",
#                             "border": "none",
#                             "padding": "10px 20px",
#                             "text-align": "center",
#                             "text-decoration": "none",
#                             "display": "inline-block",
#                             "font-size": "16px",
#                             "margin": "4px 2px",
#                             "cursor": "pointer",
#                             "border-radius": "4px",
#                         },
#                     ),
#                     dcc.Loading(
#                         id="example_similarity_matrix_loading",
#                         type="default",
#                         children=[
#                             dcc.Graph(
#                                 id="example_similarity_matrix", figure=go.Figure()
#                             ),
#                         ],
#                     ),
#                     html.Div(id="similarity_status"),
#                 ],
#                 style={
#                     "display": "flex",
#                     "justify-content": "center",
#                     "align-items": "center",
#                 },
#             ),
#             html.H3(
#                 "Instance --> Example Similarity Bar Chart",
#                 style={"text-align": "center"},
#             ),
#             html.Div(
#                 children=[
#                     dcc.Dropdown(
#                         id="example_token_comparison",
#                         multi=True,
#                         placeholder="Select Token Id...",
#                         style={
#                             "width": "300px",
#                             "margin": "4px 2px",
#                         },
#                     ),
#                     html.Button(
#                         "Compare Example Similarity",
#                         id="compare_example_similarity",
#                         n_clicks=0,
#                         style={
#                             "background-color": "#3DAFA8",
#                             "color": "white",
#                             "border": "none",
#                             "padding": "10px 20px",
#                             "text-align": "center",
#                             "text-decoration": "none",
#                             "display": "inline-block",
#                             "font-size": "16px",
#                             "margin": "4px 2px",
#                             "cursor": "pointer",
#                             "border-radius": "4px",
#                         },
#                     ),
#                     dcc.Loading(
#                         id="example_compare_similarity_loading",
#                         type="default",
#                         children=[
#                             dcc.Graph(
#                                 id="example_compare_similarity", figure=go.Figure()
#                             ),
#                         ],
#                     ),
#                     html.Div(id="compare_status"),
#                 ],
#                 style={
#                     "display": "flex",
#                     "justify-content": "center",
#                     "align-items": "center",
#                 },
#             ),
#         ],
#     )
