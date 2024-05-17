from enum import Enum

LANGUAGES = ['afr', 'eng', 'esp', 'hin', 'mar', 'pan', 'all']

FULL_LANGUAGE_NAME = {
    'afr': 'Afrikaans',
    'eng': 'English',
    'esp': 'Spanish',
    'hin': 'Hindi',
    'mar': 'Marathi',
    'pan': 'Punjabi',
    'all': 'All Languages'
}

SENTENCE_SEPARATOR = '\n'

SENTENCE_TRANSFORMERS = {
    'paraphrase multilingual miniLM': 'paraphrase-multilingual-MiniLM-L12-v2',  # all
    'mBERT': 'google-bert/bert-base-multilingual-cased',                        # all
    'XLMR': 'FacebookAI/xlm-roberta-base',                                      # all
    'LaBSE': 'sentence-transformers/LaBSE',                                     # all
}

TOKEN_TRANSFORMERS = {
    'mBERT': 'google-bert/bert-base-multilingual-cased',                        # all
    'XLMR': 'FacebookAI/xlm-roberta-base',                                      # all
    'LaBSE': 'sentence-transformers/LaBSE',                                     # all
}


class Verbose(Enum):
    SILENT = 0      # no printing
    DEFAULT = 1     # informative printing
    EXPRESSIVE = 2  # detailed printing


class EarlyStoppingOptions(Enum):
    NONE = 0  # no early stopping
    LOSS = 1  # validation loss
    CORR = 2  # correlation
