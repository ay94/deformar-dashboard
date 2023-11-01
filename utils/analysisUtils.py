from . import json, torch, pd, pkl, nn, px, AutoModel, \
    distance, np, defaultdict


class FileHandler():
    def __init__(self, project_folder: str) -> object:
        self.project_folder = project_folder

    def create_filename(self, file_name):
        return f'{self.project_folder}/{file_name}'

    def cr_fn(self, file_name):
        return self.create_filename(file_name)

    def keys_to_int(self, data):
        for key in data.keys():
            wrong_dict = data[key]['inv_labels']
            correct_dict = {int(k): v for k, v in wrong_dict.items()}
            data[key]['inv_labels'] = correct_dict

    def save_json(self, path, data):
        with open(self.cr_fn(path), 'w') as outfile:
            json.dump(data, outfile)

    def load_json(self, path):
        with open(self.cr_fn(path)) as json_file:
            data = json.load(json_file)
            self.keys_to_int(data)
            return data

    def save_object(self, obj, obj_name):
        with open(self.cr_fn(obj_name), 'wb') as output:  # Overwrites any existing file.
            pkl.dump(obj, output, pkl.HIGHEST_PROTOCOL)

    def load_object(self, obj_name):
        with open(self.cr_fn(obj_name), 'rb') as inp:
            obj = pkl.load(inp)
        return obj

    def save_model_state(self, model, model_name):
        torch.save(model.state_dict(), self.cr_fn(model_name))

    def load_model_state(self, model, model_name):
        model.load_state_dict(torch.load(self.cr_fn(model_name)))
        return model

    def save_model(self, model, model_name):
        torch.save(model, self.cr_fn(model_name))

    def load_model(self, model_name):
        model = torch.load(self.cr_fn(model_name))
        model.eval()
        return model

    def read_json(self, path):
        with open(self.cr_fn(path)) as json_file:
            data = json.load(json_file)
        return data



class TokenAmbiguity:
    def __init__(self, subwords):
        self.subwords = subwords

    def extract_token_tag_pair(self, tokens):
        pairs = []
        if len(tokens) > 1:
            print()
            for token in tokens:
                for token_tag in self.subwords[token]:
                    pairs.append((token, token_tag['tag']))
        else:
            for token_tag in self.subwords[tokens[0]]:
                pairs.append((tokens[0], token_tag['tag']))
        return pairs

    def visualize_ambiguity(self, tokens):
        token_tag_pairs = self.extract_token_tag_pair(tokens)
        # Create a dictionary of word-tag frequency counts
        word_tag_dict = {}
        for token, tag in token_tag_pairs:
            if token not in word_tag_dict:
                word_tag_dict[token] = {tag: 1}
            else:
                if tag not in word_tag_dict[token]:
                    word_tag_dict[token][tag] = 1
                else:
                    word_tag_dict[token][tag] += 1

        # Create a dataframe of word-tag frequency counts
        df = pd.DataFrame.from_dict(word_tag_dict, orient='index')

        df.fillna(value=0, inplace=True)
        # Create a heatmap using Plotly
        fig = px.imshow(df.T.values,
                        x=df.index,
                        y=df.columns,
                        text_auto=True,
                        color_continuous_scale='YlOrBr')

        fig.update_layout(
            title='Token Ambiguity Heatmap (Word Frequencies)',
            xaxis=dict(title='Tokens'),
            yaxis=dict(title='Tags'),
        )
        return fig


class TCDataset:
    def __init__(self, texts, tags, label_list, config, tokenizer, preprocessor=None):
        self.texts = texts
        self.tags = tags
        self.label_map = {label: i for i, label in enumerate(label_list)}
        self.config = config
        self.TOKENIZER = tokenizer
        self.PREPROCESSOR = preprocessor

        # Use cross entropy ignore_index as padding label id so that only real label ids contribute to the loss later.
        self.pad_token_label_id = nn.CrossEntropyLoss().ignore_index

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        textlist = self.texts[item]
        tags = self.tags[item]

        tokens = []
        label_ids = []
        word_ids = []
        for word_id, (word, label) in enumerate(zip(textlist, tags)):
            if self.PREPROCESSOR != None:
                clean_word = self.PREPROCESSOR.preprocess(word)
                word_tokens = self.TOKENIZER.tokenize(clean_word)
            else:
                word_tokens = self.TOKENIZER.tokenize(word)
                # ignore words that are preprocessed because the preprocessor return '' and the tokeniser replace that with empty list which gets ignored here
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([self.label_map[label]] + [self.pad_token_label_id] * (len(word_tokens) - 1))
                word_ids.extend([word_id] * (len(word_tokens)))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = self.TOKENIZER.num_special_tokens_to_add()
        if len(tokens) > self.config.MAX_SEQ_LEN - special_tokens_count:
            tokens = tokens[: (self.config.MAX_SEQ_LEN - special_tokens_count)]
            label_ids = label_ids[: (self.config.MAX_SEQ_LEN - special_tokens_count)]
            word_ids = word_ids[: (self.config.MAX_SEQ_LEN - special_tokens_count)]

        # Add the [SEP] token
        tokens += [self.TOKENIZER.sep_token]
        label_ids += [self.pad_token_label_id]
        token_type_ids = [0] * len(tokens)
        word_ids += [self.pad_token_label_id]

        # Add the [CLS] TOKEN
        tokens = [self.TOKENIZER.cls_token] + tokens
        label_ids = [self.pad_token_label_id] + label_ids
        token_type_ids = [0] + token_type_ids
        word_ids = [self.pad_token_label_id] + word_ids

        input_ids = self.TOKENIZER.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)
        sentence_num = [item] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = self.config.MAX_SEQ_LEN - len(input_ids)

        input_ids += [self.TOKENIZER.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length
        label_ids += [self.pad_token_label_id] * padding_length
        sentence_num += [self.pad_token_label_id] * padding_length
        word_ids += [self.pad_token_label_id] * padding_length

        assert len(input_ids) == self.config.MAX_SEQ_LEN
        assert len(attention_mask) == self.config.MAX_SEQ_LEN
        assert len(token_type_ids) == self.config.MAX_SEQ_LEN
        assert len(label_ids) == self.config.MAX_SEQ_LEN
        assert len(sentence_num) == self.config.MAX_SEQ_LEN
        assert len(word_ids) == self.config.MAX_SEQ_LEN

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'words_ids': torch.tensor(word_ids, dtype=torch.long),
            'sentence_num': torch.tensor(sentence_num, dtype=torch.long),
        }


class FineTuneConfig:
    def __init__(self) -> None:
        self.MAX_SEQ_LEN = 256
        self.TRAIN_BATCH_SIZE = 16
        self.VALID_BATCH_SIZE = 8
        self.EPOCHS = 4
        self.SPLITS = 4
        self.LEARNING_RATE = 5e-5
        self.WARMUP_RATIO = 0.1
        self.MAX_GRAD_NORM = 1.0
        self.ACCUMULATION_STEPS = 1


class TCModel(nn.Module):
    def __init__(self, num_tag, path):
        super(TCModel, self).__init__()
        self.num_tag = num_tag
        print(f'Loading BERT Model: {path}')
        self.bert = AutoModel.from_pretrained(path, output_attentions=True, output_hidden_states=True)
        self.bert_drop = nn.Dropout(0.3)
        self.output_layer = nn.Linear(self.bert.config.hidden_size, self.num_tag)

    def loss_fn(self, output, target, mask, num_labels):
        # loss function that returns the mean
        lfn = nn.CrossEntropyLoss()
        # loss function that returns the losss fore each sample
        lfns = nn.CrossEntropyLoss(reduction='none')
        # mask to specify the active losses (sentence boundary) based on attention mask
        active_loss = mask.view(-1) == 1
        # this reshape the output dimension from torch.Size([16, 256, 9]) to torch.Size([4096, 9]) now the inner dimensionality match
        active_logits = output.view(-1, num_labels)
        #  the where function takes tensor of condition, tensor of x and tensor of y if the condition is true the value of x will be used in the output tensor if the condition is flase the value of y will be used
        active_labels = torch.where(
            active_loss,
            target.view(-1),
            torch.tensor(lfn.ignore_index).type_as(target))
        # average_loss
        loss = lfn(active_logits, active_labels)
        # words loss
        losses = lfns(active_logits, active_labels)
        return loss, losses

    def forward(self, input_ids, attention_mask, token_type_ids, labels, words_ids, sentence_num):
        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bert_out = self.bert_drop(output['last_hidden_state'])
        logits = self.output_layer(bert_out)
        average_loss, losses = self.loss_fn(logits, labels, attention_mask, self.num_tag)
        return {'average_loss': average_loss, 'losses': losses, 'logits': logits,
                'last_hidden_state': output['last_hidden_state'], 'hidden_states': output['hidden_states']}


class AttentionSimilarity:
    def __init__(self, device, model1, model2, tokeniser, preprocessor):
        self.device = device
        self.model1 = model1
        self.model2 = model2
        self.tokenizer = tokeniser
        self.preprocessor = preprocessor

    def compute_similarity(self, example):
        scores = []

        sentence_a = ' '.join(example)

        if self.preprocessor is None:
            inputs = self.tokenizer.encode_plus(sentence_a, return_tensors='pt', truncation=True,
                                                add_special_tokens=True)
        else:
            inputs = self.tokenizer.encode_plus(self.preprocessor.preprocess(sentence_a), truncation=True,
                                                return_tensors='pt',
                                                add_special_tokens=True)
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)


        with torch.no_grad():

            outputs = self.model1(**inputs)
            model1_att = outputs.attentions

        with torch.no_grad():
            outputs = self.model2(**inputs)
            model2_att = outputs.attentions

        model1_mat = np.array([atten[0].cpu().numpy() for atten in model1_att])
        model2_mat = np.array([atten[0].cpu().numpy() for atten in model2_att])

        layer = []
        head = []

        for i in range(12):
            for j in range(12):
                head.append(1 - distance.cosine(
                    model1_mat[i][j].flatten(),
                    model2_mat[i][j].flatten()
                ))
            layer.append(head)
            head = []
        scores.append(layer)
        return scores[0]

