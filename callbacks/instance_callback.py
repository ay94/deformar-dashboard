import re
import json
import pandas as pd
import dash
import plotly.graph_objs as go
from dash import dcc, html, no_update, callback_context

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

import pandas as pd
import re
from dash import html

class BenajebaMatcher:
    def __init__(self, raw_text):
        self.df = self._parse_benajeba_text(raw_text)

    def _parse_benajeba_text(self, raw_text):
        
        data = []
        sentence = []
        sentence_id = 0

        for line in raw_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            word, tag = parts
            if "PERS" in tag:
                tag = tag.replace("PERS", "PER")
            sentence.append({"sentence_id": sentence_id, "word": word, "tag": tag})
            
            if word == ".":
                data.extend(sentence)
                sentence = []
                sentence_id += 1

        # If the last sentence does not end in '.', still add it
        if sentence:
            data.extend(sentence)

        return pd.DataFrame(data)


    def match_token_to_sentence_ids(self, anchor_word):
        matches = self.df[self.df["word"] == anchor_word]
        return matches["sentence_id"].unique().tolist()

    def render_sentence(self, sentence_id, anchor_word):
        sentence_df = self.df[self.df["sentence_id"] == sentence_id].copy()
        sentence_df["highlight"] = sentence_df["word"] == anchor_word
        

        styled_tokens = []
        for _, row in sentence_df.iterrows():
            # styled_tokens.append(html.Span(
            #         row["word"],
            #         style={
            #             "backgroundColor": "#FFB6A1" if row["highlight"] else "#F0F0F0",
            #             "borderRadius": "6px",
            #             "padding": "4px 6px",
            #             "margin": "2px",
            #             "display": "inline-block",
            #             "color": "#000",
            #             "fontWeight": "bold" if row["highlight"] else "normal",
            #         }
            #     ),
                                 
            # )
            styled_tokens.append(
                html.Span([
                    html.Span(
                        row["word"],
                        style={
                            "backgroundColor": "#FFB6A1" if row["highlight"] else "#F0F0F0",
                            "borderRadius": "6px",
                            "padding": "4px 6px",
                            "margin": "2px",
                            "display": "inline-block",
                            "color": "#000",
                            "fontWeight": "bold" if row["highlight"] else "normal",
                        }
                    ),
                    html.Span(
                        row["tag"],
                        style={
                            "display": "block",  # Forces tag to go *below* word
                            "fontSize": "12px",
                            "color": "#666",
                            "textAlign": "center",
                            "marginTop": "2px"
                        }
                    )
                ], style={"display": "inline-block", "textAlign": "center", "marginRight": "6px"})
            )

        return html.Div([
            html.Div("📝 Benajeba Sentence:", style={"fontWeight": "bold", "marginBottom": "8px"}),
            html.Div(styled_tokens, style={
                "direction": "rtl",
                "textAlign": "right",
                "lineHeight": "2em",
                "padding": "8px",
            })
        ])

def parse_benajeba_text(raw_text):
    data = []
    sentence_id = 0
    for block in raw_text.strip().split('\n\n'):
        for line in block.strip().split('\n'):
            if not line.strip(): continue
            word, tag = line.strip().split()
            data.append({"sentence_id": sentence_id, "word": word, "tag": tag})
        sentence_id += 1
    return pd.DataFrame(data)


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

with open("/Users/ay227/Library/CloudStorage/GoogleDrive-ahmed.younes.sam@gmail.com/My Drive/Final Year Experiments/Thesis-Experiments/RawData/ANERcorp-CamelLabSplits/ANERCorp_Benajiba.txt", "r", encoding="utf-8") as f:
        benajeba_text = f.read()
matcher = BenajebaMatcher(benajeba_text)

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
        print(variant)
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
        [
            Input("compute_token_analysis", "n_clicks"),
            Input("core_token_selector", "value"),
        ],
        [
            State("variant_selector", "value"),
            State("instance_selector", "value"),
            
        ],
        prevent_initial_call=True
    )
    def compute_token_similarity_analysis(n_clicks, token_selector_id, variant, sentence_id):
        ctx = callback_context
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        # Case 1: Token changed — reset everything
        if trigger == "core_token_selector":
            empty_fig = go.Figure(layout={"annotations": [{
                "text": "ℹ️ Select a token and click the button to compute similarity.",
                "xref": "paper", "yref": "paper",
                "showarrow": False,
                "font": {"size": 14, "color": "gray"}
            }]})

            empty_table = html.Div(
                "No similarity data yet. Click the button to compute.",
                style={"fontStyle": "italic", "color": "#888", "marginTop": "10px"}
            )

            return empty_fig, empty_fig, empty_table, empty_table
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
    [
        Output("token_label_distribution", "figure"),
        Output("token_confidence_scores", "figure"),
    ],
        Input("core_token_selector", "value"),
        State("variant_selector", "value"),
        prevent_initial_call=True
    )
    def update_label_distribution(token_selector_id, variant):
        if not token_selector_id or not variant:
            raise PreventUpdate
        try:
            label_dist_fig = tab_manager.compute_token_label_distribution(
                variant=variant,
                token_selector_id=token_selector_id
            )
            token_prediction_score_fig = tab_manager.compute_token_prediction_scores(
                variant=variant,
                token_selector_id=token_selector_id
            )
            return label_dist_fig, token_prediction_score_fig
        except Exception as e:
            print(f"❌ Failed to generate label distribution plot: {e}")
            return go.Figure(), go.Figure()
    
    

    @app.callback(
        [
            Output("token_view_split_selector", "options"),
            Output("token_view_split_selector", "value"),
            Output("token_view_sentence_selector", "options"),
            Output("token_view_sentence_selector", "value"),
        ],
        [
            Input("core_token_selector", "value"),
            Input("token_view_split_selector", "value")
        ],
        State("variant_selector", "value"),
        prevent_initial_call=True
    )
    def update_split_and_sentence(token_selector_id, split, variant):
        if not token_selector_id or not variant:
            raise PreventUpdate

        ctx = callback_context
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        anchor_token, _, _ = token_selector_id.split("@#")
        tab_data = tab_manager.get_tab_data(variant)
        train_df = tab_data.train_data
        test_df = tab_data.analysis_data

        split_options = []
        split_value = None
        sentence_options = []
        sentence_value = None

        if not train_df.empty and anchor_token in train_df["Core Tokens"].values:
            split_options.append({"label": "Train", "value": "train"})

        if not test_df.empty and anchor_token in test_df["Core Tokens"].values:
            split_options.append({"label": "Test", "value": "test"})

        # If triggered by core_token_selector → set split list, reset sentence
        if triggered_id == "core_token_selector":
            if len(split_options) == 1:
                split_value = split_options[0]["value"]
                df = train_df if split_value == "train" else test_df
                sentence_matches = df[df["Core Tokens"] == anchor_token]["Sentence Ids"].unique()
                sentence_options = [{"label": f"Sentence {int(i)}", "value": int(i)} for i in sentence_matches]
                sentence_value = int(sentence_matches[0]) if len(sentence_matches) else None
            else:
                return split_options, None, [], None

        # If triggered by split dropdown → update sentence dropdown
        elif triggered_id == "token_view_split_selector":
            df = train_df if split == "train" else test_df
            sentence_matches = df[df["Core Tokens"] == anchor_token]["Sentence Ids"].unique()
            sentence_options = [{"label": f"Sentence {int(i)}", "value": int(i)} for i in sentence_matches]
            sentence_value = int(sentence_matches[0]) if len(sentence_matches) else None
            split_value = split  # preserve split value

        return split_options, split_value, sentence_options, sentence_value



    
    @app.callback(
        Output("token_sentence_render", "children"),
        [
            Input("token_view_sentence_selector", "value"),
            Input("token_view_split_selector", "value"),
            Input("core_token_selector", "value"),
        ],
        [
            
            State("variant_selector", "value"),
        ],
        prevent_initial_call=True
    )
    def render_token_sentence(sentence_id, split, token_selector_id, variant):
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update

        trigger_id = get_input_trigger(ctx)
        # Show neutral message if triggered by token change
        if trigger_id == "core_token_selector" and (sentence_id is None or split is None):
            return html.Div(
                "👉 Please select a data split and a sentence to view the token in context.",
                style={
                    "fontStyle": "italic",
                    "color": "#888",
                    "marginTop": "10px"
                }
            )
        tab_data = tab_manager.get_tab_data(variant)
        if not split:
            return no_update
        dataset = tab_data.train_data if split == "train" else tab_data.analysis_data
        
        
        
        if not token_selector_id:
            return no_update
        anchor_token, _, token_index = token_selector_id.split("@#")
        token_index = int(token_index)
        
        # if split == 'test':
        sentence_df = dataset[dataset["Sentence Ids"] == sentence_id]
        core_sentence_df = sentence_df[sentence_df['Labels'] != -100]
        # Determine anchor row
        anchor_row = core_sentence_df[core_sentence_df["Core Tokens"] == anchor_token]
        if anchor_row.empty:
            return html.Div("⚠️ Could not find the token to highlight in the selected sentence.")

        anchor_word = anchor_row["Words"].values[0]
        is_arabic = bool(re.search(r'[\u0600-\u06FF]', anchor_word))
        direction = "rtl" if is_arabic else "ltr"
        text_align = "right" if is_arabic else "left"

        # Add highlight column
        core_sentence_df = core_sentence_df.copy()
        core_sentence_df["highlight"] = core_sentence_df["Core Tokens"] == anchor_token
        # Render the sentence with highlighted word
        styled_tokens = []
        for _, row in core_sentence_df.iterrows():
            word = row["Words"]
            tag = row["True Labels"]
            highlight = row["highlight"]

        #     styled_tokens.append(html.Span(
        #         word,
        #         style={
        #             "backgroundColor": "#FFB6A1" if highlight else "#F0F0F0",
        #             "borderRadius": "6px",
        #             "padding": "4px 6px",
        #             "margin": "2px",
        #             "display": "inline-block",
        #             "color": "#000",
        #             "fontWeight": "bold" if highlight else "normal",
        #         }
        #     ))

        # return html.Div([
        #     html.Div("📝 Sentence:", style={"fontWeight": "bold", "marginBottom": "8px"}),
        #     html.Div(styled_tokens, style={
        #         "direction": direction,
        #         "textAlign": text_align,
        #         "lineHeight": "2em",
        #         "padding": "8px",
        #     })
        # ])
            styled_tokens.append(
                    html.Span([
                        html.Span(
                            word,
                            style={
                                "backgroundColor": "#FFB6A1" if highlight else "#F0F0F0",
                                "borderRadius": "6px",
                                "padding": "4px 6px",
                                "margin": "2px",
                                "display": "inline-block",
                                "color": "#000",
                                "fontWeight": "bold" if highlight else "normal",
                            }
                        ),
                        html.Span(
                            tag,
                            style={
                                "display": "block",  # Forces tag to go *below* word
                                "fontSize": "12px",
                                "color": "#666",
                                "textAlign": "center",
                                "marginTop": "2px"
                            }
                        )
                    ], style={"display": "inline-block", "textAlign": "center", "marginRight": "6px"})
                )

        return html.Div([
            html.Div("📝 Benajeba Sentence:", style={"fontWeight": "bold", "marginBottom": "8px"}),
            html.Div(styled_tokens, style={
                "direction": "rtl",
                "textAlign": "right",
                "lineHeight": "2em",
                "padding": "8px",
            })
        ])
    

    @app.callback(
        Output("token_origin_sentence", "options"),
        Input("core_token_selector", "value"),
        State("variant_selector", "value"),
        prevent_initial_call=True
    )
    def populate_benajeba_sentences(token_selector_id, variant):
        if not token_selector_id:
            return no_update
        tab_data = tab_manager.get_tab_data(variant)
        analysis_data = tab_data.analysis_data
        word = analysis_data[analysis_data['Token Selector Id'] == token_selector_id]['Words'].unique()[0]
        
        
        
        
        
        sentence_ids = matcher.match_token_to_sentence_ids(word)

        options = [{"label": f"Benajeba Sentence {sid}", "value": sid} for sid in sentence_ids]
        return options
    
    @app.callback(
        Output("token_origin_sentence_render", "children"),
        [
            Input("token_origin_sentence", "value"),
            State("core_token_selector", "value"),
            State("variant_selector", "value"),
        ],
        prevent_initial_call=True
    )
    def render_benajeba_sentence(sentence_id, token_selector_id, variant):
        if not sentence_id or not token_selector_id:
            return no_update

        tab_data = tab_manager.get_tab_data(variant)
        analysis_data = tab_data.analysis_data
        word = analysis_data[analysis_data['Token Selector Id'] == token_selector_id]['Words'].unique()[0]
        return matcher.render_sentence(sentence_id, word)



        
        
        