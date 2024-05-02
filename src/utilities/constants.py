from enum import Enum

LANGUAGES = ['afr', 'eng', 'esp', 'hin', 'mar', 'pan']

FULL_LANGUAGE_NAME = {
    'afr': 'Afrikaans',
    'eng': 'English',
    'esp': 'Spanish',
    'hin': 'Hindi',
    'mar': 'Marathi',
    'pan': 'Punjabi'
}

SENTENCE_SEPARATOR = '\n'

SENTENCE_TRANSFORMERS = {
    'all MiniLM': 'all-MiniLM-L6-v2',
    'paraphrase multilingual miniLM': 'paraphrase-multilingual-MiniLM-L12-v2',
    'mBERT': 'google-bert/bert-base-multilingual-cased',
    'XLMR': 'FacebookAI/xlm-roberta-base',
    'LaBSE': 'sentence-transformers/LaBSE',
    'ALBETO': 'dccuchile/albert-base-spanish',
    'BETO': 'dccuchile/bert-base-spanish-wwm-cased',
    'RoBERTa-BNE': 'PlanTL-GOB-ES/roberta-base-bne'
}

TOKEN_TRANSFORMERS = {
    'base uncased BERT': 'bert-base-uncased',
    'mBERT': 'google-bert/bert-base-multilingual-cased',
    'XLMR': 'FacebookAI/xlm-roberta-base',
    'LaBSE': 'sentence-transformers/LaBSE',
    'ALBETO': 'dccuchile/albert-base-spanish',
    'BETO': 'dccuchile/bert-base-spanish-wwm-cased',
    'RoBERTa-BNE': 'PlanTL-GOB-ES/roberta-base-bne'
}


class Verbose(Enum):
    SILENT = 0
    DEFAULT = 1
    EXPRESSIVE = 2
