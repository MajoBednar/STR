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
    'all MiniLM': 'all-MiniLM-L6-v2',                                           # eng
    'paraphrase multilingual miniLM': 'paraphrase-multilingual-MiniLM-L12-v2',  # all
    'mBERT': 'google-bert/bert-base-multilingual-cased',                        # all
    'XLMR': 'FacebookAI/xlm-roberta-base',                                      # all
    'LaBSE': 'sentence-transformers/LaBSE',                                     # all
    'ALBETO': 'dccuchile/albert-base-spanish',                                  # esp
    'BETO': 'dccuchile/bert-base-spanish-wwm-cased',                            # esp
    'RoBERTa-BNE': 'PlanTL-GOB-ES/roberta-base-bne'                             # esp
}

TOKEN_TRANSFORMERS = {
    'base uncased BERT': 'bert-base-uncased',                                   # eng
    'mBERT': 'google-bert/bert-base-multilingual-cased',                        # all
    'XLMR': 'FacebookAI/xlm-roberta-base',                                      # all
    'LaBSE': 'sentence-transformers/LaBSE',                                     # all
    'ALBETO': 'dccuchile/albert-base-spanish',                                  # esp
    'BETO': 'dccuchile/bert-base-spanish-wwm-cased',                            # esp
    'RoBERTa-BNE': 'PlanTL-GOB-ES/roberta-base-bne'                             # esp
}


class Verbose(Enum):
    SILENT = 0
    DEFAULT = 1
    EXPRESSIVE = 2
