import re
import logging
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import plotly.express as px
import numpy as np
from tqdm import tqdm
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from config.config_managers import ColorMap
from config.enums import CorrelationColumns
from bertviz import head_view, model_view
from managers.plotting.decision_plotting_managers import (
    CentroidAverageSimilarity, CorrelationMatrix, DecisionScatter,
    MeasureScatter, SelectionTagProportion, SimilarityMatrix, TrainScatter)
from managers.tabs.tab_managers import BaseTabManager
from dash import html
from experiment_utils.analysis import AttentionSimilarity
import torch
from transformers import AutoModelForTokenClassification

class InstanceTabManager(BaseTabManager):
    def __init__(self, variants_data):
        super().__init__(variants_data)
        self.loaded_models = {}  # Tracks loaded models per variant

    def load_models(self, variant):
        data = self.get_tab_data(variant)
        if data:
            try:
                # Load pretrained and fine-tuned models
                _ = data.get_pretrained_model
                _ = data.get_fine_tuned_model

                # Load dataset manager and datasets
                _ = data.get_dataset_manager
                _ = data.get_train_dataset
                _ = data.get_test_dataset

                self.loaded_models[variant] = True
                return True
            except Exception as e:
                print(f"Model loading failed for {variant}: {e}")

        self.loaded_models[variant] = False
        return False


    def is_model_loaded(self, variant):
        return self.loaded_models.get(variant, False)
    
    def get_sentence_ids(self, variant, selected_ids=None):
        
        tab_data = self.get_tab_data(variant)
        if not tab_data or tab_data.analysis_data.empty:
            return None  # Let the callback handle PreventUpdate

        df = tab_data.analysis_data
        if 'Sentence Ids' not in df.columns:
            return None
        if selected_ids:
           df = df[df['Global Id'].isin(selected_ids)]

        return df['Sentence Ids'].unique().tolist()
    
    def get_instance_row(self, variant, sentence_id):
        tab_data = self.get_tab_data(variant)
        if not tab_data or tab_data.analysis_data.empty:
            return None

        df = tab_data.analysis_data
        row = df[df["Sentence Ids"] == sentence_id]
        

        if row.empty:
            return None

        # Assuming only one row per sentence
        row = row.iloc[0]

        return {
            "Tokens": row["Tokens"],
            "True Labels": row["True Labels"],
            "Pred Labels": row["Pred Labels"]
        }
    
    def get_sentence_slice(self, variant, instance_id):
        df = self.get_tab_data(variant).analysis_data
        sentence_df = df[df["Sentence Ids"] == instance_id]
        # Extract tokens and labels
        words_df = sentence_df[sentence_df['Labels'] != -100]
        words = words_df['Words']
        word_true_labels = words_df['True Labels']
        tokens = sentence_df["Tokens"].tolist()
        true_labels = sentence_df["True Labels"].tolist()
        pred_labels = sentence_df["Pred Labels"].tolist()
        core_tokens = words_df["Tokens"].tolist()
        core_mistake_labels = words_df["Confusion Components"].tolist()
        return words, word_true_labels, tokens, true_labels, pred_labels, core_tokens, core_mistake_labels
    
    def generate_instance_output(self, variant, instance_id):
        words, word_true_labels, tokens, true_labels, pred_labels, core_tokens, core_mistake_labels = self.get_sentence_slice(variant, instance_id)
        color_util = ColorMap()
        # ➕ Detect if Arabic (for RTL support)
        is_arabic = any(re.search(r'[\u0600-\u06FF]', word) for word in words)
        direction = "rtl" if is_arabic else "ltr"
        text_align = "right" if is_arabic else "left"
        
        sentence_colored = InstanceTabManager.generate_colored_tokens(words, word_true_labels, color_util.color_map)
        truth_colored = InstanceTabManager.generate_colored_tokens(tokens, true_labels, color_util.color_map)
        pred_colored = InstanceTabManager.generate_colored_tokens(tokens, pred_labels, color_util.color_map)
        mistake_colored = InstanceTabManager.generate_colored_tokens(core_tokens, core_mistake_labels, color_util.color_map)
        # ➕ Wrap in a styled Div that uses the correct direction
        wrapper_style = {
            "direction": direction,
            "textAlign": text_align,
            "lineHeight": "2em",
            "padding": "8px"
        }

        return (
            html.Div(sentence_colored, style=wrapper_style),
            html.Div(truth_colored, style=wrapper_style),
            html.Div(pred_colored, style=wrapper_style),
            html.Div(mistake_colored, style=wrapper_style)
        )
    
   
    def get_entity_level_annotations_non_strict(self, variant, sentence_id):
        tab_data = self.get_tab_data(variant)
        if not tab_data or tab_data.analysis_data.empty:
            return [], [], {}, []

        df = tab_data.analysis_data
        core_data = df[df["Labels"] != -100]
        sentence_df = core_data[core_data["Sentence Ids"] == sentence_id]

        if sentence_df.empty:
            return [], [], {}, []

        y_true = sentence_df["True Labels"].tolist()
        y_pred = sentence_df["Pred Labels"].tolist()
        words = sentence_df["Words"].tolist()

        analyzer = NonStrictEntityAnalyzer(sentence_df)
        true_entities = analyzer.extract_entities([y_true])[0]
        pred_entities = analyzer.extract_entities([y_pred])[0]
        error_dict = self.get_entity_errors(true_entities, pred_entities)

        return true_entities, pred_entities, error_dict, words
    
    
    def get_entity_level_annotations_strict(self, variant, sentence_id):
        tab_data = self.get_tab_data(variant)
        if not tab_data or tab_data.analysis_data.empty:
            return [], [], {}, []

        df = tab_data.analysis_data
        core_data = df[df["Labels"] != -100]
        sentence_df = core_data[core_data["Sentence Ids"] == sentence_id]

        if sentence_df.empty:
            return [], [], {}, []

        y_true = sentence_df["True Labels"].tolist()
        y_pred = sentence_df["Pred Labels"].tolist()
        words = sentence_df["Words"].tolist()

        analyzer = StrictEntityAnalyzer(sentence_df)
        true_entities = analyzer.extract_entities([y_true]).entities[0]
        pred_entities = analyzer.extract_entities([y_pred]).entities[0]
        error_dict = self.get_entity_errors(true_entities, pred_entities)

        return true_entities, pred_entities, error_dict, words
    
    
    def render_entity_tags(self, entities, words, error_dict=None):
        if not entities:
            return [html.Span("No Errors", style={
                "color": ColorMap().color_map["No Errors"],
                "fontWeight": "bold"
            })]

        error_dict = error_dict or {}
        fp_set = set(error_dict.get("FP", []))
        fn_set = set(error_dict.get("FN", []))

        color_util = ColorMap()
        tags = []
        # Detect Arabic for layout direction
        is_arabic = any(re.search(r'[\u0600-\u06FF]', word) for word in words)
        direction = "rtl" if is_arabic else "ltr"
        text_align = "right" if is_arabic else "left"
        for ent in entities:
            _, etype, start, end = ent
            label = f"{etype} [{' '.join(words[start:end+1])}]"

            # Determine base color
            base_color = color_util.color_map.get(etype, "#ccc")

            # Override for errors
            if ent in fp_set:
                background = color_util.color_map.get("FP", "#EF553B")
                border = f"2px solid {background}"
            elif ent in fn_set:
                background = color_util.color_map.get("FN", "#00CC96")
                border = f"2px dashed {background}"
            else:
                background = base_color
                border = "2px solid transparent"

            style = {
                "backgroundColor": background,
                "color": "#fff",
                "padding": "6px 10px",
                "margin": "6px",
                "borderRadius": "6px",
                "display": "inline-block",
                "fontWeight": "bold",
                "fontSize": "0.85rem",
                "border": border,
                "direction": direction,
                "textAlign": text_align
            }

            tags.append(html.Span(label, style=style))

        return tags


    def generate_attention_analysis(self, variant, sentence_id, view_option):
        if not self.is_model_loaded(variant):
            logging.warning(f"❌ Model not loaded for variant: {variant}")
            # return "", "", go.Figure()
            if not self.load_models(variant):
                logging.error(f"❌ Could not auto-load model for {variant}")
                return "", "", go.Figure()

        try:
            data = self.get_tab_data(variant)
            df = data.analysis_data
            model_pre = data.get_pretrained_model
            model_fine = data.get_fine_tuned_model
            dataset = data.get_test_dataset
            tokenizer = dataset.tokenizer
            preprocessor = dataset.preprocessor
        except Exception as e:
            logging.error(f"⚠️ Failed to load components for {variant}: {e}")
            return "", "", go.Figure()

        if sentence_id is None:
            logging.warning("⚠️ Sentence ID is None — falling back to first instance")
            sentence_id = 0

        try:
            example = dataset.__getitem__(int(sentence_id))
        except Exception as e:
            logging.error(f"⚠️ Failed to get item {sentence_id} from dataset: {e}")
            example = dataset.__getitem__(0)

        try:
            input_ids = example['input_ids'][example['input_ids'] != 0]
            attention_mask = example['attention_mask'][example['input_ids'] != 0]
            token_type_ids = example['token_type_ids'][example['input_ids'] != 0]

            tokens = tokenizer.convert_ids_to_tokens(input_ids)

            input_ids = input_ids.unsqueeze(0)
            mask = attention_mask.unsqueeze(0)
            type_ids = token_type_ids.unsqueeze(0)

            with torch.no_grad():
                pre_out = model_pre(
                    input_ids=input_ids,
                    attention_mask=mask,
                    token_type_ids=type_ids,
                    # output_attentions=True  # 👈 ADD THIS
                )

                fin_out = model_fine.bert(
                    input_ids=input_ids,
                    attention_mask=mask,
                    token_type_ids=type_ids,
                    # output_attentions=True  # 👈 ADD THIS
                )


            pre_attention = pre_out[-1]
            fin_attention = fin_out[-1]

            if view_option == "head":
                pre_vis = head_view(pre_attention, tokens, html_action='return')
                fin_vis = head_view(fin_attention, tokens, html_action='return')
            elif view_option == "model":
                pre_vis = model_view(pre_attention, tokens, html_action='return')
                fin_vis = model_view(fin_attention, tokens, html_action='return')
            else:
                logging.warning(f"⚠️ Invalid view option: {view_option}")
                return "", "", go.Figure()

            # Attention similarity heatmap
            attn_sim = AttentionSimilarity(
                device=torch.device("cpu"),
                model1=model_pre,
                model2=model_fine.bert,
                tokenizer=tokenizer,
                preprocessor=preprocessor,
            )

            sentence_words = df[(df['Sentence Ids'] == int(sentence_id))&(df['Labels']!=-100)]['Words'].tolist()
            similarity_scores = attn_sim.compute_similarity(sentence_words)

            heatmap = px.imshow(
                similarity_scores,
                labels={"x": "Heads", "y": "Layers", "color": "Similarity"},
                color_continuous_scale="RdBu_r",
                title="Attention Similarity Between Pretrained and Fine-tuned",
            )
            heatmap.update_layout(
                autosize=False,
                width=1000,
                height=600,
                margin=dict(l=50, r=50, t=50, b=50),
            )

            logging.info(f"✅ Attention analysis generated for sentence {sentence_id} [{variant}]")

            return pre_vis.data, fin_vis.data, heatmap

        except Exception as e:
            logging.error(f"❌ Failed during attention analysis for {variant}, sentence {sentence_id}: {e}")
            return "", "", go.Figure()
     
    def compute_token_similarity_analysis(self, variant, token_selector_id, split="train", top_k=20):
        try:
            anchor_token, token_position, sentence_id  = token_selector_id.split("@#")
            sentence_id = int(sentence_id)
            token_position = int(token_position)
        except Exception as e:
            logging.error(f"❌ Invalid Token Selector ID format: {token_selector_id} | Error: {e}")
            return go.Figure(), pd.DataFrame()

        logging.info(f"🔍 Running token similarity for: '{anchor_token}' in sentence {sentence_id} (token position  {token_position}) [split: {split}]")

        tab_data = self.get_tab_data(variant)
        if not tab_data or not self.is_model_loaded(variant):
            logging.warning("⚠️ Variant not loaded or model unavailable.")
            if not self.load_models(variant):
                logging.error("❌ Model could not be loaded for token analysis.")
                return go.Figure(), pd.DataFrame()

        # Get model & dataset
        model = tab_data.get_fine_tuned_model.bert
        tokenizer = tab_data.get_test_dataset.tokenizer

        dataset = tab_data.get_train_dataset if split == "train" else tab_data.get_test_dataset
        df = tab_data.train_data if split == "train" else tab_data.analysis_data

        # Get anchor representation
        instance = dataset.__getitem__(sentence_id)
        token_input = {
            "input_ids": instance["input_ids"].unsqueeze(0),
            "attention_mask": instance["attention_mask"].unsqueeze(0),
            "token_type_ids": instance.get("token_type_ids", torch.zeros_like(instance["input_ids"])).unsqueeze(0)
        }

        with torch.no_grad():
            output = model(**token_input, output_hidden_states=False)
            anchor_vector = output.last_hidden_state[0, token_position].cpu().numpy()

        # Filter similar tokens in that split with same text
        candidates = df[
            (df["Labels"] != -100) &
            (df["Core Tokens"] == anchor_token)
        ][["Global Id", "Words", "Sentence Ids", "Token Positions"]]
        

        logging.info(f"📌 Found {len(candidates)} candidates matching token '{anchor_token}' in {split} split")
        candidate_sample = candidates.sample(min(20, len(candidates)))
        logging.info(f"Plotting only {len(candidate_sample)}")


        results = []

        for _, row in tqdm(candidate_sample.iterrows()):
            try:
                sen_id = int(row["Sentence Ids"])
                tok_pos = int(row["Token Positions"])
                word = row["Words"]
                global_id = row["Global Id"]

                example = dataset.__getitem__(sen_id)
                inputs = {
                    "input_ids": example["input_ids"].unsqueeze(0),
                    "attention_mask": example["attention_mask"].unsqueeze(0),
                    "token_type_ids": example.get("token_type_ids", torch.zeros_like(example["input_ids"])).unsqueeze(0)
                }

                with torch.no_grad():
                    rep_output = model(**inputs, output_hidden_states=False)
                    candidate_vector = rep_output.last_hidden_state[0, tok_pos].cpu().numpy()

                sim_score = cosine_similarity([anchor_vector], [candidate_vector])[0][0]

                token_id = example["input_ids"][tok_pos].item()
                token_str = tokenizer.convert_ids_to_tokens([token_id])[0]

                results.append({
                    "Token": token_str,
                    "Word": word,
                    "Sentence ID": sen_id,
                    "Split": split,
                    "Similarity": sim_score
                })
            except Exception as e:
                logging.warning(f"⚠️ Skipped token due to error: {e}")
                continue

        if not results:
            logging.warning("❌ No similar tokens found.")
            return go.Figure(), pd.DataFrame()
        
        table_data = pd.DataFrame(results)
        table_data["Similarity"] = table_data["Similarity"].astype(float).round(5)
        result_df = table_data.sort_values("Similarity", ascending=False).head(top_k)
        result_df["Sentence"] = "Sentence " + result_df["Sentence ID"].astype(str)


        fig = px.bar(
            result_df,
            x="Similarity",
            y="Token",
            color="Sentence",
            hover_data=["Word", "Sentence ID"],
            title=f"Top {top_k} Similar Tokens in {split.title()} Split",
            template="plotly_white",
            barmode="group",              # Group bars per sentence
            text="Similarity",            # Show similarity value on bar
        )

        # Optional: Make text clean and rotate x labels
        # fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_traces(orientation='h')  # horizontal bars
        fig.update_layout(
            xaxis_title="Similarity",
            yaxis_title="Token",
            width=600,     # or 600
            height=400,    # increase if needed
            )



        return fig, table_data
    
    def compute_token_label_distribution(self, variant, token_selector_id):
        try:
            anchor_token, _, _ = token_selector_id.split("@#")
        except Exception as e:
            logging.error(f"❌ Invalid Token Selector ID format: {token_selector_id} | Error: {e}")
            return go.Figure()

        tab_data = self.get_tab_data(variant)
        if not tab_data:
            return go.Figure()

        # Get train/test dataframes
        train_df = tab_data.train_data
        test_df = tab_data.analysis_data

        # Filter rows where the core token matches
        train_counts = (
            train_df[train_df["Core Tokens"] == anchor_token]["True Labels"]
            .value_counts()
            .rename_axis("Label")
            .reset_index(name="Count")
        )
        train_counts["Split"] = "Train"

        test_counts = (
            test_df[test_df["Core Tokens"] == anchor_token]["True Labels"]
            .value_counts()
            .rename_axis("Label")
            .reset_index(name="Count")
        )
        test_counts["Split"] = "Test"

        # Combine for grouped bar chart
        combined = pd.concat([train_counts, test_counts])

        fig = px.bar(
            combined,
            x="Label",
            y="Count",
            color="Split",
            barmode="group",
            title=f"Label Distribution for Token '{anchor_token}'",
            template="plotly_white"
        )

        fig.update_layout(
            xaxis_title="Entity Label",
            yaxis_title="Count",
            legend_title="Data Split"
        )

        return fig
    
    def compute_token_prediction_scores(self, variant, token_selector_id):
        try:
            anchor_token, _, _ = token_selector_id.split("@#")
        except Exception as e:
            logging.error(f"❌ Invalid Token Selector ID format: {token_selector_id} | Error: {e}")
            return go.Figure()

        tab_data = self.get_tab_data(variant)
        if not tab_data:
            return go.Figure()

       
        analysis_data = tab_data.analysis_data
        token_df = analysis_data[analysis_data['Token Selector Id'] == token_selector_id]
        columns = CorrelationColumns()
        confidence_metrics = columns.confidence_metrics
        # Melt (reshape) the dataframe to long format
        melted_df = token_df.melt(
            value_vars=confidence_metrics,
            var_name='Entity Tags',
            value_name='Confidence Score'
        )

        fig = px.bar(
            melted_df,
            x="Entity Tags",
            y="Confidence Score",
            # color="Split",
            # barmode="group",
            title=f"Confidence Scores by Entity Type",
            template="plotly_white"
        )

        fig.update_layout(
            xaxis_title="Entity Label",
            yaxis_title="Count",
            legend_title="Data Split"
        )

        return fig



    
    @staticmethod
    def get_entity_errors(true_entities, pred_entities):
        true_set = set(true_entities)
        pred_set = set(pred_entities)

        false_positives = list(pred_set - true_set)
        false_negatives = list(true_set - pred_set)

        return {"FP": false_positives, "FN": false_negatives}
            
    @staticmethod
    def generate_colored_tokens(tokens, labels, color_map, show_label=True):
        """
        Generate a list of html.Span components with background color based on labels.

        Args:
            tokens (list[str]): The tokenized words.
            labels (list[str]): Corresponding labels for the tokens.
            color_map (dict): Label to color hex mapping.
            show_label (bool): Whether to show the label text on the token.

        Returns:
            list[html.Span]: Dash components for visual display.
        """
        styled_tokens = []
        for token, label in zip(tokens, labels):
            
            if label=='TN':
                continue  # 🚫 Skip True Negatives entirely
            if token in ['[CLS]', '[SEP]']:
                continue  # 🚫 Skip True Negatives entirely
            background_color = color_map.get(label, "#E0E0E0")
            token_display = f"{token} ({label})" if show_label and label != "O" else token

            styled_tokens.append(
                html.Span(
                    token_display,
                    style={
                        "backgroundColor": background_color,
                        "borderRadius": "6px",
                        "padding": "4px 6px",
                        "margin": "2px",
                        "display": "inline-block",
                        "color": "#fff" if label != "O" else "#000",
                        "fontWeight": "bold",
                    }
                )
            )
        return styled_tokens
    
        






from dataclasses import dataclass, field
from typing import List, Dict, Set
from abc import ABC, abstractmethod
from collections import defaultdict
from seqeval.scheme import Entities, IOB2, IOB1
from seqeval.metrics.sequence_labeling import get_entities
import pandas as pd
pd.set_option("display.max_rows", 200)  # Display all rows

class EntityErrorAnalyzer(ABC):
    """Abstract base class for entity analysis."""

    def __init__(self, df):
        self.df = df
        self.y_true, self.y_pred = self.prepare_data(df)
        self.true_entities = []
        self.pred_entities = []

    @abstractmethod
    def extract_entities(self, y_data):
        """Extract entities based on the specific mode (strict or non-strict)."""
        pass

    @abstractmethod
    def prepare_entities(self):
        """Prepare true and predicted entities for analysis."""
        pass
    
    def prepare_data(self, df):
        core_data = df[df['Labels'] !=-100]
        y_true = core_data.groupby('Sentence Ids')['True Labels'].apply(list).tolist()
        y_pred = core_data.groupby('Sentence Ids')['Pred Labels'].apply(list).tolist()
        return y_true, y_pred
    
    def compute_false_negatives(self, entity_type):
        """Compute false negatives for a specific entity type."""
        return set(
            [e for e in self.true_entities if e[1] == entity_type]
        ) - set([e for e in self.pred_entities if e[1] == entity_type])

    def compute_false_positives(self, entity_type):
        """Compute false positives for a specific entity type."""
        return set(
            [e for e in self.pred_entities if e[1] == entity_type]
        ) - set([e for e in self.true_entities if e[1] == entity_type])

    def analyze_sentence_errors(self, target_entities, comparison_entities):
        """Analyze errors and return sentence IDs by error type."""
        error_sentences = defaultdict(set)  # Dictionary to hold sentence IDs for each error type
        non_o_errors = set()
        indexed_entities = defaultdict(list)

        # Index comparison entities by sentence
        for entity in comparison_entities:
            sen, entity_type, start, end = entity
            indexed_entities[sen].append(entity)

        # First pass: entity errors
        for target_entity in target_entities:
            t_sen, t_type, t_start, t_end = target_entity

            for comp_entity in indexed_entities[t_sen]:
                c_type, c_start, c_end = comp_entity[1:]

                if (
                    t_start == c_start
                    and t_end == c_end
                    and t_type != c_type
                    and target_entity not in non_o_errors
                ):
                    non_o_errors.add(target_entity)
                    error_sentences["Entity"].add(target_entity)

        # Second pass: boundary errors
        for target_entity in target_entities - non_o_errors:
            t_sen, t_type, t_start, t_end = target_entity

            for comp_entity in indexed_entities[t_sen]:
                c_sen, c_type, c_start, c_end = comp_entity

                if (
                    t_type == c_type
                    and (t_start <= c_start <= t_end or t_start <= c_end <= t_end)
                    and target_entity not in non_o_errors
                ):
                    non_o_errors.add(target_entity)
                    error_sentences["Boundary"].add(target_entity)

        # Third pass: combined entity and boundary errors
        for target_entity in target_entities - non_o_errors:
            t_sen, t_type, t_start, t_end = target_entity

            for comp_entity in indexed_entities[t_sen]:
                c_sen, c_type, c_start, c_end = comp_entity

                if (
                    c_type != t_type
                    and (t_start <= c_start <= t_end or t_start <= c_end <= t_end)
                    and target_entity not in non_o_errors
                ):
                    non_o_errors.add(target_entity)
                    error_sentences["Entity and Boundary"].add(target_entity)
                    # print(t_sen, t_start, t_end, c_sen, c_start, c_end)
                    # print(f' ({t_start} <= {c_start} <= {t_end} or {t_start} <= {c_end} <= {t_end})')
                    

        # Remaining unmatched errors are "O errors"
        for target_entity in target_entities - non_o_errors:
            t_sen, t_type, t_start, t_end = target_entity
            error_sentences["O"].add(target_entity)

        return {error_type: list(s_ids) for error_type, s_ids in error_sentences.items()}


    def analyze_component(self, error_type, entity_type=None):
        
        """Analyze errors (FP or FN) for a specific or all entity types."""
        self.prepare_entities()
        error_analysis = {}
        entity_types = (
            [entity_type]
            if entity_type
            else set(e[1] for e in self.true_entities + self.pred_entities)
        )

        for etype in entity_types:
            if error_type == "false_negatives":
                target_entities = self.compute_false_negatives(etype)
            elif error_type == "false_positives":
                target_entities = self.compute_false_positives(etype)
            else:
                raise ValueError("Error type must be 'false_negative' or 'false_positive'.")

            error_analysis[etype] = self.analyze_sentence_errors(
                target_entities, self.pred_entities if error_type == "false_negatives" else self.true_entities
            )

        return error_analysis
    
    def analyze_errors(self):
        self.prepare_entities()
        """Analyze both false positives and false negatives."""
        error_components = {"false_positives": defaultdict(set), "false_negatives": defaultdict(set)}

        for error_component in error_components.keys():
            results = self.analyze_component(error_component)
            for entity_type, errors in results.items():
                for error_type, sentences in errors.items():
                    error_components[error_component][error_type].update(sentences)

        # Convert sets to lists for consistency
        return {k: {etype: set(ids) for etype, ids in v.items()} for k, v in error_components.items()}
    
    


class StrictEntityAnalyzer(EntityErrorAnalyzer):
    """Analyzer for strict entity processing."""

    def extract_entities(self, y_data):
        """Extract entities in strict mode."""
        entities = Entities(y_data, IOB2, False)
        return self.adjust_end_index(entities)

    def prepare_entities(self):
        """Prepare true and predicted entities for strict mode."""
        self.true_entities = self.flatten_entities(self.extract_entities(self.y_true))
        self.pred_entities = self.flatten_entities(self.extract_entities(self.y_pred))

    def print_sentence(self, sen_id):
        """Print entities for a specific sentence ID."""
        true_entities = self.extract_entities(self.y_true).entities
        pred_entities = self.extract_entities(self.y_pred).entities
        print(f"True: {true_entities[sen_id]}")
        print(f"Pred: {pred_entities[sen_id]}")
        error = set(pred_entities[sen_id]) - set(true_entities[sen_id])
        print(f"Error in Pred: {error}")
        core_data = self.df[self.df['Labels'] !=-100]
        sentence_data = core_data[core_data['Sentence Ids']  == sen_id].copy()
        print(sentence_data[['Words', 'Sentence Ids', 'True Labels', 'Pred Labels', 'Strict True Entities', 'Strict Pred Entities', 'True Entities', 'Pred Entities']].head(60).to_string())

    @staticmethod
    def flatten_entities(entities):
        """Flatten strict entities into tuples."""
        return [e for sen in entities.entities for e in sen]
    
    @staticmethod
    def adjust_end_index(entities):
        """Adjust the end index for IOB2 entities to make them inclusive."""
        adjusted_entities = []
        for sentence_entities in entities.entities:  # Iterate through sentences
            adjusted_sentence = []
            for entity in sentence_entities:  # Iterate through entities in each sentence
                sentence_id, entity_type, start, end = entity.to_tuple()
                # Adjust end index
                adjusted_sentence.append((sentence_id, entity_type, start, end - 1))
            adjusted_entities.append(adjusted_sentence)
        entities.entities = adjusted_entities  # Replace with adjusted entities
        return entities
    
    
    
    
class NonStrictEntityAnalyzer(EntityErrorAnalyzer):
    """Analyzer for non-strict entity processing."""

    def extract_entities(self, y_data):
        """Extract entities in non-strict mode."""
        return [
            [(sen_id,) + entity for entity in get_entities(sen)]
            for sen_id, sen in enumerate(y_data)
        ]

    def prepare_entities(self):
        """Prepare true and predicted entities for non-strict mode."""
        self.true_entities = self.flatten_entities(self.extract_entities(self.y_true))
        self.pred_entities = self.flatten_entities(self.extract_entities(self.y_pred))

    def print_sentence(self, sen_id):
        """Print entities for a specific sentence ID."""
        true_entities = self.extract_entities(self.y_true)
        pred_entities = self.extract_entities(self.y_pred)
        print(f"True: {true_entities[sen_id]}")
        print(f"Pred: {pred_entities[sen_id]}")
        error = set(pred_entities[sen_id]) - set(true_entities[sen_id])
        print(f"Error in Pred: {error}")
        core_data = self.df[self.df['Labels'] !=-100]
        sentence_data = core_data[core_data['Sentence Ids']  == sen_id].copy()
        print(sentence_data[['Words', 'Sentence Ids', 'True Labels', 'Pred Labels', 'Strict True Entities', 'Strict Pred Entities', 'True Entities', 'Pred Entities']].head(60).to_string())
        
    @staticmethod
    def flatten_entities(entities):
        """Flatten non-strict entities into tuples."""
        return [e for sen in entities for e in sen]

class ErrorAnalysisManager:
    """Manages all error analysis workflows and stores results."""

    def __init__(self, df):
        """
        Initialize the manager with the dataset.

        Args:
            df (pd.DataFrame): The dataset containing y_true and y_pred.
        """
        self.df = df
        self.strict_analyzer = StrictEntityAnalyzer(df)
        self.non_strict_analyzer = NonStrictEntityAnalyzer(df)
        self.results = {
            "IOB2": {"false_negatives": None, "false_positives": None, "errors": None},
            "IOB": {"false_negatives": None, "false_positives": None, "errors": None},
        }

    def run_workflows(self):
        """Run all error analysis workflows."""
        self.results["IOB2"]["false_negatives"] = self.strict_analyzer.analyze_component("false_negatives")
        self.results["IOB2"]["false_positives"] = self.strict_analyzer.analyze_component("false_positives")
        self.results["IOB2"]["errors"] = self.strict_analyzer.analyze_errors()

        self.results["IOB"]["false_negatives"] = self.non_strict_analyzer.analyze_component("false_negatives")
        self.results["IOB"]["false_positives"] = self.non_strict_analyzer.analyze_component("false_positives")
        self.results["IOB"]["errors"] = self.non_strict_analyzer.analyze_errors()

    def get_results(self):
        """Get the results of all workflows."""
        return self.results

class SchemeComparator:
    """Facilitator for comparing annotation schemes."""

    def __init__(self, results):
        """
        Initialize the comparator with results from error analysis.

        Args:
            results (dict): Results from the manager's workflows, structured by scheme.
        """
        self.results = results

    def compare_component(self, component, entity_type):
        """
        Compare all error types for a specific entity across schemes.

        Args:
            entity_type (str): The entity type to compare (e.g., "MISC").

        Returns:
            dict: A dictionary with set operation results for all error types.
        """
        schemes = list(self.results.keys())
        if len(schemes) != 2:
            raise ValueError("Comparator requires exactly two schemes for comparison.")

        scheme_1, scheme_2 = schemes
        component_1 = self.results[scheme_1][component]
        component_2 = self.results[scheme_2][component]

        results = {}
        entity_1 = component_1.get(entity_type, {})
        entity_2 = component_2.get(entity_type, {})

        # Compare all error types under the given entity
        all_error_types = set(entity_1.keys()).union(set(entity_2.keys()))
        for error_type in all_error_types:
            set_1 = set(entity_1.get(error_type, []))
            set_2 = set(entity_2.get(error_type, []))

            results[error_type] = {
                "overlap": set_1 & set_2,
                f"{scheme_1} Only": set_1 - set_2,
                f"{scheme_2} Only": set_2 - set_1,
            }

        return results

    def compare_errors(self, component, error_type):
        """
        Compare errors across all entities and error types for both schemes.

        Returns:
            dict: A dictionary with set operation results for all error types.
        """
        schemes = list(self.results.keys())
        if len(schemes) != 2:
            raise ValueError("Comparator requires exactly two schemes for comparison.")

        schemes_map = {'scheme_1': 'IOB', 'scheme_2': 'IOB2'}
        errors_1 = self.results[schemes_map['scheme_1']]["errors"][component]
        errors_2 = self.results[schemes_map['scheme_2']]["errors"][component]

       
       
        comparison_result = ComparisonResult.from_lists(errors_1, errors_2, error_type, schemes_map)

        return comparison_result.to_dict()


@dataclass
class ComparisonResult:
    """Dataclass to store comparison results."""
    scheme_1_name: str
    scheme_2_name: str
    set_1_errors: Set[int] = field(default=set)
    set_2_errors: Set[int] = field(default=set)
    overlap: Set[int] = field(default_factory=set)
    scheme_1_only: Set[int] = field(default_factory=set)
    scheme_2_only: Set[int] = field(default_factory=set)

    @staticmethod
    def from_lists(errors_1: Dict, errors_2: Dict, error_type: str, schemes_map: Dict) -> "ComparisonResult":
        """
        Create a ComparisonResult from two lists.

        Args:
            lst_1: List of values from scheme 1.
            lst_2: List of values from scheme 2.

        Returns:
            ComparisonResult: Dataclass containing the comparison and statistics.
        """
        set_1 = set(errors_1.get(error_type, []))
        
        set_2 = set(errors_2.get(error_type, []))
        
        sentence_lst_1 = [error[0] for error in errors_1.get(error_type, [])]
        sentence_lst_2 = [error[0] for error in errors_2.get(error_type, [])]
        sentence_set_1 = set(sentence_lst_1)
        sentence_set_2 = set(sentence_lst_2)
        
        overlap = sentence_set_1 & sentence_set_2
        scheme_1_only = sentence_set_1 - sentence_set_2
        scheme_2_only = sentence_set_2 - sentence_set_1

        return ComparisonResult(
            scheme_1_name=schemes_map['scheme_1'],
            scheme_2_name=schemes_map['scheme_2'],
            set_1_errors= set_1,
            set_2_errors= set_2,
            overlap=overlap,
            scheme_1_only=scheme_1_only,
            scheme_2_only=scheme_2_only,
        )
        
    def to_dict(self) -> Dict[str, Dict[str, Set[int]]]:
        """R"Overlap": self.overlap, comparison results as a dictionary."""
        return {
            f"{self.scheme_1_name} Errors": self.set_1_errors,
            f"{self.scheme_2_name} Errors": self.set_2_errors,
            "Overlap": self.overlap,
            f"{self.scheme_1_name} Only Errors": self.scheme_1_only,
            f"{self.scheme_2_name} Only Errors": self.scheme_2_only,
        }
