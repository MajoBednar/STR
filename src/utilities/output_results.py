from .constants import FULL_LANGUAGE_NAME as FULL


def print_results(model_name: str, language: str, spearman_correlation: float):
    print(f'Model:                {model_name}')
    print(f'Language:             {FULL[language]}')
    print(f'Spearman Correlation: {spearman_correlation}')
