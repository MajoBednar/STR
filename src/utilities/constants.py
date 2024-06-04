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
    'miniLM': 'paraphrase-multilingual-MiniLM-L12-v2',
    'mBERT': 'google-bert/bert-base-multilingual-cased',
    'XLMR': 'FacebookAI/xlm-roberta-base',
    'LaBSE': 'sentence-transformers/LaBSE',
}

TOKEN_TRANSFORMERS = {
    'mBERT': 'google-bert/bert-base-multilingual-cased',
    'XLMR': 'FacebookAI/xlm-roberta-base',
    'LaBSE': 'sentence-transformers/LaBSE',
}


class Verbose(Enum):
    SILENT = 0      # no printing
    DEFAULT = 1     # informative printing
    EXPRESSIVE = 2  # detailed printing


class EarlyStoppingOptions(Enum):
    NONE = 0  # no early stopping
    LOSS = 1  # validation loss
    CORR = 2  # correlation
