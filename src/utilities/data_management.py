import pandas as pd
from scipy.stats import spearmanr

from .constants import SENTENCE_SEPARATOR_FOR_LANGAUGE as SEPARATOR, FULL_LANGUAGE_NAME as FULL


def load_scores(df: pd.DataFrame) -> list[float]:
    scores = df["Score"].tolist()
    scores = [float(score) for score in scores]
    return scores


def load_sentence_pairs(df: pd.DataFrame, language: str) -> list[list[str]]:
    sentences = df["Text"].tolist()
    sentence_pairs = []
    for sentence in sentences:
        sentence_1, sentence_2 = sentence.split(SEPARATOR[language])
        sentence_pairs.append([sentence_1, sentence_2])
    return sentence_pairs


def load_data(language: str, dataset: str) -> tuple[list[float], list[list[str]]]:
    data_path = 'data/datasets_original_splits/' + language + '/' + language + dataset + '.csv'
    df = pd.read_csv(data_path)
    scores = load_scores(df)
    sentence_pairs = load_sentence_pairs(df, language)
    return scores, sentence_pairs


class DataManager:
    def __init__(self, language):
        self.language: str = language

        data = self.initialize_data()
        self.scores_train: list[float] = data[0]
        self.scores_dev: list[float] = data[1]
        self.scores_test: list[float] = data[2]

        self.sentence_pairs_train: list[list[str]] = data[3]
        self.sentence_pairs_dev: list[list[str]] = data[4]
        self.sentence_pairs_test: list[list[str]] = data[5]

        self.spearman_correlation: float = 0

    def initialize_data(self) -> tuple:
        scores_train, sentence_pairs_train = load_data(language=self.language, dataset='_train')
        scores_dev, sentence_pairs_dev = load_data(language=self.language, dataset='_dev_with_labels')
        scores_test, sentence_pairs_test = load_data(language=self.language, dataset='_test_with_labels')
        return scores_train, scores_dev, scores_test, sentence_pairs_train, sentence_pairs_dev, sentence_pairs_test

    def calculate_spearman_correlation(self, true_scores, predicted_scores):
        self.spearman_correlation, _ = spearmanr(true_scores, predicted_scores)

    def print_results(self, model_name: str, dataset: str = 'Test') -> None:
        print()
        print(f'Model:                {model_name}')
        print(f'Language:             {FULL[self.language]}')
        print(f'Set:                  {dataset}')
        print(f'Spearman Correlation: {self.spearman_correlation}')
        print()
