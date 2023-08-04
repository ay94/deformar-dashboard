import copy
import os
import time
import torch
import pandas as pd
import plotly.io as pio
from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor

from utils.analysisUtils import TokenAmbiguity, FineTuneConfig, TCDataset, TCModel
from . import copy, os, time, torch, pd, pio, AutoModel, AutoTokenizer, \
    ArabertPreprocessor, defaultdict

Datasets = {
    'ANERCorp_CamelLab': {
        'model_name': {'arabertv02': 'aubmindlab/bert-base-arabertv02',
                       'mbert': 'bert-base-multilingual-cased'},
        'splits': {'Train': 'train', 'Val': 'val', 'Test': 'test'}
    },

    'conll2003': {
        'model_name': {'bert_cased': 'bert-base-cased',
                       'mbert': 'bert-base-multilingual-cased'},
        'splits': {'Train': 'train', 'Val': 'val', 'Test': 'test'}
    },

}


class DatasetConfig:

    def __init__(self):
        self.file_handler = None
        self.dataset_name = None
        self.model_name = None
        self.model_path = None
        self.split = None
        self.finetuned = None
        self.instanceLevel = None
        self.pretrained = None
        self.loaded = False
        self.initialized = False


    def load_data(self, fh, dataset_name, model_name, model_path, split):


        start_time = time.time()
        self.created = self.created_(fh.cr_fn(f'{dataset_name}/initialization'))
        self.file_handler = fh
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.model_path = model_path
        self.split = split

        corpora = self.file_handler.load_json(f'corpora.json')
        self.corpus = corpora[self.dataset_name]

        train_subwords = self.defaultify((self.file_handler.read_json(f'{self.dataset_name}/train_subwords.json')))
        self.token_ambiguity = TokenAmbiguity(train_subwords)

        analysis_df = pd.read_json(
            self.file_handler.cr_fn(f'{self.dataset_name}/{self.split}/{self.split}_analysis_df.jsonl.gz'),
            lines=True
        )
        token_score = pd.read_json(
            self.file_handler.cr_fn(f'{self.dataset_name}/{self.split}/{self.split}_token_score_df.jsonl.gz'),
            lines=True
        )
        self.analysis_df = self.clean_analysis_df(analysis_df, token_score)

        if self.split != 'train':
            self.light_train_df = pd.read_json(
                self.file_handler.cr_fn(f'{self.dataset_name}/light_train_df.jsonl.gz'),
                lines=True
            )
        else:
            self.light_train_df = self.analysis_df.copy()[["token_ids", "words", "agreement", "truth", "pred", "x", "y"]]

        self.centroid_df = pd.read_json(
            self.file_handler.cr_fn(f'{self.dataset_name}/{self.split}/{self.split}_centroid_df.jsonl.gz'),
            lines=True
        )

        self.confusion_data = pd.read_json(
            self.file_handler.cr_fn(f'{self.dataset_name}/{self.split}/{self.split}_confusion_data.jsonl.gz'),
            lines=True
        )

        self.entity_prediction = pd.read_json(
            self.file_handler.cr_fn(f'{self.dataset_name}/{self.split}/{self.split}_entity_prediction.jsonl.gz'),
            lines=True
        )

        self.seq_report = pd.read_csv(
            self.file_handler.cr_fn(f'{self.dataset_name}/{self.split}/{self.split}_seq_report.csv')
        )
        self.skl_report = pd.read_csv(
            self.file_handler.cr_fn(f'{self.dataset_name}/{self.split}/{self.split}_skl_report.csv')
        )

        self.light_df = self.create_view_df()

        self.weights, self.activations = self.read_plotly()

        self.dataset_end_time = (time.time() - start_time) / 60
        self.loaded = True

    def clean_analysis_df(self, analysis_df, token_score):
        analysis_df['sen_id'] = analysis_df['sen_id'].astype(str)
        analysis_df['word_id'] = analysis_df['word_id'].astype(str)
        analysis_df.set_index('global_id', inplace=True)
        token_score.set_index('global_id', inplace=True)
        analysis_df.loc[token_score.index, ['truth_token_score', 'pred_token_score']] = token_score[
            ['truth_token_score', 'pred_token_score']]
        analysis_df.reset_index(inplace=True)
        return analysis_df

    def create_view_df(self):
        selected_columns = ['global_id', 'sen_id', 'token_ids', 'first_tokens', 'words', 'agreement', 'truth', 'pred',
                            "error_type", 'x', 'y', 'confidences', 'variability', 'prediction_entropy', 'token_entropy']
        light_df = self.analysis_df[selected_columns].copy()
        light_df['id'] = light_df['global_id']
        light_df.set_index('id', inplace=True, drop=False)
        return light_df

    def read_plotly(self):
        # Read the JSON string from the file using pandas
        with open(self.file_handler.cr_fn(f'{self.dataset_name}/{self.split}/{self.split}_weights.json'), 'r') as file:
            weights_json = file.read()

        with open(self.file_handler.cr_fn(f'{self.dataset_name}/{self.split}/{self.split}_activations.json'),
                  'r') as file:
            activations_json = file.read()
        # Create a Plotly figure from the JSON string
        weights = pio.from_json(weights_json)
        activations = pio.from_json(activations_json)
        return weights, activations

    def create_model(self):
        start_time = time.time()
        self.created = self.created_(self.file_handler.cr_fn(f'{self.dataset_name}/initialization'))
        self.finetuned = TCModel(len(self.corpus['labels']), self.model_path)
        self.pretrained = copy.deepcopy(self.finetuned.bert)
        self.finetuned.load_state_dict(torch.load(
            self.file_handler.cr_fn(
                f'{self.dataset_name}/initialization/{self.model_name}_{self.dataset_name}_regular_state.bin'),
            map_location=torch.device('cpu')))

        tokenizer = AutoTokenizer.from_pretrained(self.model_path, do_lower_case=False)
        preprocessor = ArabertPreprocessor(self.model_path)
        self.pretrained_bert = AutoModel.from_pretrained(self.model_path, output_attentions=True,
                                                         output_hidden_states=True)

        config = FineTuneConfig()
        train_dataset = TCDataset(
            texts=[x[1] for x in self.corpus['train'][:]],
            tags=[x[2] for x in self.corpus['train'][:]],
            label_list=self.corpus['labels'],
            config=config,
            tokenizer=tokenizer,
            preprocessor=preprocessor)
        val_dataset = TCDataset(
            texts=[x[1] for x in self.corpus['val'][:]],
            tags=[x[2] for x in self.corpus['val'][:]],
            label_list=self.corpus['labels'],
            config=config,
            tokenizer=tokenizer,
            preprocessor=preprocessor)

        test_dataset = TCDataset(
            texts=[x[1] for x in self.corpus['test'][:]],
            tags=[x[2] for x in self.corpus['test'][:]],
            label_list=self.corpus['labels'],
            config=config,
            tokenizer=tokenizer,
            preprocessor=preprocessor)

        self.instanceLevel = InstanceLevel(tokenizer, preprocessor, train_dataset, val_dataset, test_dataset)
        self.pretrained_bert.save_pretrained(
            self.file_handler.cr_fn(
                f'{self.dataset_name}/initialization/pretrained_{self.model_name}_{self.dataset_name}_regular.bin')
        )
        self.file_handler.save_object(self.instanceLevel, f'{self.dataset_name}/initialization/instanceLevel.pkl')

        self.create_end_time = (time.time() - start_time) / 60

    def initialize_model(self):
        start_time = time.time()
        self.created = self.created_(self.file_handler.cr_fn(f'{self.dataset_name}/initialization'))
        self.instanceLevel = self.file_handler.load_object(f'{self.dataset_name}/initialization/instanceLevel.pkl')
        self.pretrained_bert = AutoModel.from_pretrained(
            self.file_handler.cr_fn(
                f'{self.dataset_name}/initialization/pretrained_{self.model_name}_{self.dataset_name}_regular.bin')
        )
        self.finetuned = TCModel(len(self.corpus['labels']), self.model_path)
        self.finetuned.load_state_dict(torch.load(
            self.file_handler.cr_fn(
                f'{self.dataset_name}/initialization/{self.model_name}_{self.dataset_name}_regular_state.bin'),
            map_location=torch.device('cpu')))

        self.initialize_end_time = (time.time() - start_time) / 60
        self.initialized = True

    def created_(self, path):
        file_list = os.listdir(path)
        if len([file_ for file_ in file_list if file_ != '.DS_Store']) > 1:
            # The directory has files
            return True
        else:
            # The directory is empty
            return False

    def defaultify(self, dictionary):
        output = defaultdict(list)
        for key, values in dictionary.items():
            for value in values:
                output[key].append(value)
        return output



class InstanceLevel:
    def __init__(self, tokenizer, preprocessor, train_dataset, val_dataset, test_dataset):
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
