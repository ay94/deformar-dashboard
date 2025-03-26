import re
import json
import pandas as pd
import dash
import plotly.graph_objs as go
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from layouts.managers.layout_managers import (get_input_trigger,
                                              process_selection, render_basic_table_with_font)
from managers.tabs.instance_tab_managers import InstanceTabManager
DISPLAY_COLUMNS = [
        "Global Id", "Sentence Ids", "Words", "Tokens", "Token Selector Id",  
        "Token Ambiguity", "Word Ambiguity", "Consistency Ratio",
        "Inconsistency Ratio", "Tokenization Rate", "Token Confidence",
        "Loss Values", "Prediction Uncertainty", "True Silhouette", "Pred Silhouette",
    ]

from dash import dash_table

def render_similarity_table(df):
    return dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#3DAFA8",
            "fontWeight": "bold",
            "color": "white"
        },
        style_cell={
            "textAlign": "left",
            "padding": "8px",
            "minWidth": "100px",
            "maxWidth": "300px",
            "whiteSpace": "normal",
        },
        page_size=10
    )


def register_callbacks(app, variants_data):
    tab_manager = InstanceTabManager(variants_data)
    @app.callback(
        [
            Output("instance_selector", "options"),
            Output("impact_instances", "options"),
        ],
        [
            Input("variant_selector", "value"),
            Input("decision_store", "data"),  
            Input("measure_store", "data"),
        ]
    )
    def update_instance_dropdowns(variant, decision_selection, measure_selection):
        if not variant:
            raise PreventUpdate
        selected_ids = process_selection(decision_selection or measure_selection)
        sentence_ids = tab_manager.get_sentence_ids(variant, selected_ids)
        if sentence_ids is None:
            raise PreventUpdate

        options = [{"label": f"Sentence {idx}", "value": idx} for idx in sentence_ids]
        return options, options
    
    @app.callback(
        [
            Output("instance_sentence", "children"),
            Output("instance_truth", "children"),
            Output("instance_pred", "children"),
            Output("instance_mistakes", "children"),
        ],
        [
            Input("variant_selector", "value"),
            Input("instance_selector", "value"),
        ]
    )
    def update_instance_display(variant, instance_id):
        if not variant or instance_id is None:
            raise PreventUpdate
        sentence_colored, truth_colored, pred_colored, mistake_colored = tab_manager.generate_instance_output(variant, instance_id)
        return sentence_colored, truth_colored, pred_colored, mistake_colored
    
    

    @app.callback(
        [
            Output("entity_true_iob", "children"),
            Output("entity_pred_iob", "children"),
            Output("entity_error_iob", "children"),
            Output("entity_true_iob2", "children"),
            Output("entity_pred_iob2", "children"),
            Output("entity_error_iob2", "children"),
        ],
        [
            Input("variant_selector", "value"),
            Input("instance_selector", "value"),
        ],
    )
    def update_entity_annotations(variant, instance_id):
        if not variant or instance_id is None:
            raise PreventUpdate

        true_iob, pred_iob, errors_iob, words_iob = tab_manager.get_entity_level_annotations_non_strict(variant, instance_id)
        rendered_true_iob = tab_manager.render_entity_tags(true_iob, words_iob, errors_iob)
        rendered_pred_iob = tab_manager.render_entity_tags(pred_iob, words_iob, errors_iob)
        rendered_error_iob = tab_manager.render_entity_tags(errors_iob["FP"] + errors_iob["FN"], words_iob, errors_iob)

        true_iob2, pred_iob2, errors_iob2, words_iob2 = tab_manager.get_entity_level_annotations_strict(variant, instance_id)
        rendered_true_iob2 = tab_manager.render_entity_tags(true_iob2, words_iob2, errors_iob2)
        rendered_pred_iob2 = tab_manager.render_entity_tags(pred_iob2, words_iob2, errors_iob2)
        rendered_error_iob2 = tab_manager.render_entity_tags(errors_iob2["FP"] + errors_iob2["FN"], words_iob2, errors_iob2)

        return (
            rendered_true_iob,
            rendered_pred_iob,
            rendered_error_iob,
            rendered_true_iob2,
            rendered_pred_iob2,
            rendered_error_iob2,
        )
    
    @app.callback(
        Output("model_status", "children"),
        Input("load_model_btn", "n_clicks"),
        State("variant_selector", "value"),
        prevent_initial_call=True,
    )
    def load_models_callback(n_clicks, variant):
        if not variant:
            return "❌ Please select a variant"

        success = tab_manager.load_models(variant)

        return "✅ Model & Data Loaded" if success else "❌ Failed to Load Models"
    
    @app.callback(
        [
            Output('pre_attention_view', 'srcDoc'),
            Output('fin_attention_view', 'srcDoc'),
            Output('instance_training_impact', 'figure'),
        ],
        [
            Input("visualize_training_impact", "n_clicks"),
            State("variant_selector", "value"),
            State("impact_instances", "value"),
            State("attention_view", "value"),
        ],
        prevent_initial_call=True
    )
    def visualize_training_impact_callback(n_clicks, variant, sentence_id, view_option):
        return tab_manager.generate_attention_analysis(variant, sentence_id, view_option)
    
    @app.callback(
        Output("core_token_selector", "options"),
        [
            Input("variant_selector", "value"),
            Input("instance_selector", "value")
        ]
    )
    def update_core_token_dropdown(variant, instance_id):
        if not variant or instance_id is None:
            raise PreventUpdate

        tab_data = tab_manager.get_tab_data(variant)
        if not tab_data or tab_data.analysis_data.empty:
            raise PreventUpdate

        df = tab_data.analysis_data
        core_df = df[(df["Labels"] != -100) & (df["Sentence Ids"] == instance_id)]

        if core_df.empty:
            return []

        options = [
            {"label": token_id.split("@#")[0], "value": token_id}
            for token_id in core_df["Token Selector Id"].tolist()
        ]

        return options
    
   
    
    @app.callback(
        [
            
            Output("token_similarity_train", "figure"),
            Output("token_similarity_test", "figure"),
            Output("token_similarity_table_train", "children"),
            Output("token_similarity_table_test", "children"),
        ],
        Input("compute_token_analysis", "n_clicks"),
        [
            State("variant_selector", "value"),
            State("instance_selector", "value"),
            State("core_token_selector", "value"),
        ],
        prevent_initial_call=True
    )
    def compute_token_analysis(n_clicks, variant, sentence_id, token_selector_id):
        
        if not (variant and sentence_id is not None and token_selector_id is not None):
            raise PreventUpdate

        try:
            fig_train, df_train = tab_manager.compute_token_similarity_analysis(variant, token_selector_id, split="train")
            fig_test, df_test = tab_manager.compute_token_similarity_analysis(variant, token_selector_id, split="test")

            train_table = (
                render_basic_table_with_font(df_train) if not df_train.empty else html.Div("⚠️ No similar tokens found in train.")
                # dash_table.DataTable(
                #     data=df_train.to_dict("records"),
                #     columns=[{"name": col, "id": col} for col in df_train.columns],
                #     style_table={"overflowX": "auto"},
                #     style_cell={"textAlign": "left", "padding": "5px"},
                # ) 
            )

            test_table = (
                render_basic_table_with_font(df_test) if not df_test.empty else html.Div("⚠️ No similar tokens found in test.")
                # dash_table.DataTable(
                #     data=df_test.to_dict("records"),
                #     columns=[{"name": col, "id": col} for col in df_test.columns],
                #     style_table={"overflowX": "auto"},
                #     style_cell={"textAlign": "left", "padding": "5px"},
                # ) 
            
            )

            return fig_train, fig_test, train_table, test_table

        except Exception as e:
            print("❌ Token analysis failed:", str(e))
            return go.Figure(), go.Figure(), html.Div("❌ Error"), html.Div("❌ Error")
        
    
    @app.callback(
    Output("token_label_distribution", "figure"),
        Input("core_token_selector", "value"),
        State("variant_selector", "value"),
        prevent_initial_call=True
    )
    def update_label_distribution(token_selector_id, variant):
        if not token_selector_id or not variant:
            raise PreventUpdate

        try:
            fig = tab_manager.compute_token_label_distribution(
                variant=variant,
                token_selector_id=token_selector_id
            )
            return fig
        except Exception as e:
            print(f"❌ Failed to generate label distribution plot: {e}")
            return go.Figure()
    
    @app.callback(
        [
            Output("token_view_split_selector", "options"),
            Output("token_view_split_selector", "value"),
            Output("token_view_sentence_selector", "options"),
            Output("token_view_sentence_selector", "value")
        ],
        Input("core_token_selector", "value"),
        State("variant_selector", "value"),
        prevent_initial_call=True
    )
    def update_sentence_dropdown(token_selector_id, variant):
        if not token_selector_id or not variant:
            raise PreventUpdate

        anchor_token, _, _ = token_selector_id.split("@#")  # anchor_token, sent_id, token_index
        

        tab_data = tab_manager.get_tab_data(variant)
        train_df = tab_data.train_data
        test_df = tab_data.analysis_data
        
        # Filter matches
        train_matches = train_df[train_df["Core Tokens"] == anchor_token]["Sentence Ids"].unique()
        test_matches = test_df[test_df["Core Tokens"] == anchor_token]["Sentence Ids"].unique()
        

        options = []
        split_value = None
        sentence_options = []
        sentence_value = None

        if len(train_matches) and len(test_matches):
            options = [{"label": "Train", "value": "train"}, {"label": "Test", "value": "test"}]
            split_value = "train"
            sentence_options = [{"label": f"Sentence {i}", "value": i} for i in train_matches]
            sentence_value = train_matches[0]
        elif len(train_matches):
            options = [{"label": "Train", "value": "train"}]
            split_value = "train"
            sentence_options = [{"label": f"Sentence {i}", "value": i} for i in train_matches]
            sentence_value = train_matches[0]
        elif len(test_matches):
            options = [{"label": "Test", "value": "test"}]
            split_value = "test"
            sentence_options = [{"label": f"Sentence {i}", "value": i} for i in test_matches]
            sentence_value = test_matches[0]
        else:
            raise PreventUpdate

        return options, split_value, sentence_options, sentence_value
    
    
    @app.callback(
        Output("token_sentence_render", "children"),
        [
            Input("token_view_sentence_selector", "value"),
            Input("token_view_split_selector", "value"),
        ],
        [
            State("core_token_selector", "value"),
            State("variant_selector", "value"),
        ],
        prevent_initial_call=True
    )
    def render_token_sentence(sentence_id, split, token_selector_id, variant):
        if not token_selector_id or sentence_id is None or not variant or not split:
            raise PreventUpdate

        anchor_token, _, token_index = token_selector_id.split("@#")
        token_index = int(token_index)

        tab_data = tab_manager.get_tab_data(variant)
        dataset = tab_data.get_train_dataset if split == "train" else tab_data.get_test_dataset
        tokenizer = dataset.tokenizer
        example = dataset.__getitem__(int(sentence_id))

        input_ids = example['input_ids'][example['input_ids'] != 0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        styled_tokens = []
        for idx, tok in enumerate(tokens):
            highlight = tok == anchor_token
            styled_tokens.append(html.Span(
                tok,
                style={
                    "backgroundColor": "#FFB6A1" if highlight else "#F0F0F0",
                    "borderRadius": "6px",
                    "padding": "4px 6px",
                    "margin": "2px",
                    "display": "inline-block",
                    "color": "#000",
                    "fontWeight": "bold" if highlight else "normal",
                }
            ))
            is_arabic = bool(re.search(r'[\u0600-\u06FF]', anchor_token))

            direction = "rtl" if is_arabic else "ltr"
            text_align = "right" if is_arabic else "left"

        return html.Div([
            html.Div("📝 Sentence:", style={"fontWeight": "bold", "marginBottom": "8px"}),
            html.Div(styled_tokens, style={
                "direction": direction,
                "textAlign": text_align,
                "lineHeight": "2em",
                "padding": "8px",
            })
        ])

        


        
        




