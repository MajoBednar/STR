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
    'paraphrase multilingual miniLM': 'paraphrase-multilingual-MiniLM-L12-v2'
}

TOKEN_TRANSFORMERS = {
    'base uncased BERT': 'bert-base-uncased'
}

class Verbose(Enum):
    SILENT = 0
    DEFAULT = 1
    EXPRESSIVE = 2
