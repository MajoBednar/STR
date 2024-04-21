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


class Verbose(Enum):
    SILENT = 0
    DEFAULT = 1
    EXPRESSIVE = 2
