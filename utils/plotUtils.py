from . import confusion_matrix, pd
from . import html

color_map = {'B-LOC': 'darkgreen', 'B-PERS': 'deepskyblue',
             'B-ORG': 'darkcyan', 'B-MISC': 'palevioletred',
             'I-LOC': 'yellowgreen', 'I-PERS': 'lightblue',
             'I-ORG': 'cyan', 'I-MISC': 'violet', 'O': 'saddlebrown',
             'LOC': 'darkgreen', 'PERS': 'deepskyblue', 'ORG': 'darkcyan',
             'MISC': 'palevioletred', 'NOUN': 'darkgreen', 'VERB': 'deepskyblue',
             'PN': 'darkcyan', 'PRT': 'yellowgreen', 'ADJ': 'lightblue',
             'ADV': 'cyan', 'PRON': 'saddlebrown', 'DSIL': 'violet', 'CCONJ': 'turquoise',
             'ADP': 'darksalmon', 'PUNCT': 'tomato', 'DET': 'midnightblue', 'X': 'olive',
             'AUX': 'limegreen', 'NUM': 'slateblue', 'PART': 'wheat', 'SYM': 'firebrick',
             'PROPN': 'gold', 'INTJ': 'lightseagreen', 'IGNORED': 'grey', '[CLS]': 'grey',
             '[SEP]': 'grey', 'Selected': 'black'}


def get_input_trigger(ctx):
    return ctx.triggered[0]["prop_id"].split(".")[0]


def default_coordinates(x, y):
    if x is None or len(x) < 1:
        x_column = 'x'
    else:
        x_column = x[0]
    if y is None or len(y) < 1:
        y_column = 'y'
    else:
        y_column = y[0]
    return x_column, y_column


def default_color(colors):
    if colors is None or len(colors) < 1:
        color = 'truth'
        symbol = 'agreement'
    else:
        if len(colors) > 0:
            color = colors[0]
            symbol = 'agreement'
        if len(colors) > 1:
            color = colors[0]
            symbol = colors[1]

    return color, symbol


def default_entity(entity):
    if entity is None or len(entity) < 1:
        entity = 'LOC'
    else:
        entity = entity

    return entity


def defualt_centroid(centroid):
    if centroid is None or len(centroid) < 1:
        centroid = 'Centroid-9'
    else:
        centroid = centroid

    return centroid


def get_value(df, column_name, value):
    return df[df[df.columns[list(df.columns).index(column_name)]] == value]


def create_confusion_table(errors):
    confusion = errors.pivot_table(index='truth', columns='pred', aggfunc='size', fill_value=0)
    return confusion


def compute_confusion(data):
    true_labels = data['truth']
    predicted_labels = data['pred']
    # Create a confusion matrix
    labels = list(set(true_labels))
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    confusion = pd.DataFrame(cm, columns=labels)
    confusion.index = labels
    return confusion


def create_token_confusion(df):
    token_data = df[~df['truth'].isin(['[CLS]', '[SEP]', 'IGNORED'])]
    return token_data[False == token_data['agreement']]


def create_error_bars(df, entity_prediction, chosen_entity):
    errors = get_value(df, 'agreement', False)
    entity_errors = get_value(entity_prediction, 'agreement', False)
    confusion_table = create_confusion_table(errors)
    entity_errors = entity_errors.rename(columns={'true_token': 'truth', 'pred_token': 'pred'})
    entity_confusion_table = create_confusion_table(
        entity_errors[entity_errors['entity'] == chosen_entity])
    return confusion_table, entity_confusion_table


def extract_column(column):
    if column is None or len(column) < 1:
        output = 'first_tokens'
    else:
        output = column
    return output


def identify_mistakes(tokens, truth, pred):
    mistakes = []
    for i, (t, p) in enumerate(zip(truth, pred)):
        if t != p:
            mistakes.append(f'Token: ({tokens[i]}) => GS: {t} => PR: {p}')
    return html.Span(f'Number of Prediction Mistakes: {len(mistakes)} =>  {" # ".join(mistakes)}')


def get_indices(lst, element):
    indices = [index for index, value in enumerate(lst) if value == element]
    return indices


def color_tokens(example_words, example_labels, label_map, tokens, labels, preds, token=None):
    colored_words = []
    colored_truth_text = []
    colored_pred_text = []
    label_color_map = [html.Span(lb, style={'background-color': color_map[lb], 'margin-right': '5px', 'padding': '2px',
                                            'color': 'white'}) for lb in label_map]
    if token is None:
        word_locator = []
    else:
        word_locator = get_indices(example_words, token)
    for i, (word, word_lb) in enumerate(zip(example_words, example_labels)):
        if i in word_locator:
            colored_words.append(
                html.Span(word, style={'background-color': 'red', 'margin-right': '5px', 'padding': '2px',
                                       'color': 'white'}))
        else:
            colored_words.append(
                html.Span(word, style={'background-color': color_map[word_lb],
                                       'margin-right': '5px', 'padding': '2px',
                                       'color': 'white'}))
    for token, label, pred in zip(tokens, labels, preds):
        colored_truth_text.append(
            html.Span(token, style={'background-color': color_map[label],
                                    'margin-right': '5px', 'padding': '2px',
                                    'color': 'white'}))
        colored_pred_text.append(
            html.Span(token, style={'background-color': color_map[pred],
                                    'margin-right': '5px', 'padding': '2px',
                                    'color': 'white'}))
    return label_color_map, colored_words, colored_truth_text, colored_pred_text


def min_max(df, ratio):
    x_range = (
        df.x.min() - abs(df.x.min() * ratio), df.x.max() + abs(df.x.max() * ratio))
    y_range = (
        df.y.min() - abs(df.y.min() * ratio), df.y.max() + abs(df.y.max() * ratio))
    return x_range, y_range


def default_view(view):
    if view is None or len(view) < 1:
        output = 'head'
    else:
        output = view

    return output
