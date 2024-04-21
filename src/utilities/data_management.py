import pandas as pd
import os
from scipy.stats import spearmanr

from .constants import SENTENCE_SEPARATOR as SEP, FULL_LANGUAGE_NAME as FULL


def print_missing_dataset_warning(old_dataset: str, new_dataset: str) -> None:
    print(f'WARNING: {old_dataset} dataset is missing and being replaced with {new_dataset} dataset')


def replace_missing_dataset(language: str, dataset: str) -> str:
    new_dataset = 'REPLACEMENT FAILED'
    match dataset:
        case '_train' | '_test_with_labels':
            new_dataset = '_dev_with_labels'
    print_missing_dataset_warning(language + dataset, language + new_dataset)
    return language + new_dataset + '.csv'


def load_scores(df: pd.DataFrame) -> list[float]:
    scores = df['Score'].tolist()
    scores = [float(score) for score in scores]
    return scores


def load_sentence_pairs(df: pd.DataFrame) -> list[list[str]]:
    sentences = df["Text"].tolist()
    sentence_pairs = []
    for sentence in sentences:
        sentence_1, sentence_2 = sentence.split(SEP)
        sentence_pairs.append([sentence_1, sentence_2])
    return sentence_pairs


def load_data(language: str, dataset: str) -> tuple[list[float], list[list[str]], bool]:
    path = 'data/datasets_original_splits/' + language + '/'
    file = language + dataset + '.csv'
    replaced = False
    if not os.path.exists(path + file):
        file = replace_missing_dataset(language, dataset)
        replaced = True
    df = pd.read_csv(path + file)
    scores = load_scores(df)
    sentence_pairs = load_sentence_pairs(df)
    return scores, sentence_pairs, replaced


class DataManager:
    def __init__(self, language):
        self.language: str = language

        data = self.__initialize_data()
        self.scores: dict[str, list[float]] = {
            'Train': data[0],
            'Dev': data[1],
            'Test': data[2],
            'Train+Dev': data[0] + data[1]
        }
        self.sentence_pairs: dict[str, list[list[str]]] = {
            'Train': data[3],
            'Dev': data[4],
            'Test': data[5]
        }
        self.spearman_correlation: float = 0
        self.warning = data[6]

    def __initialize_data(self) -> tuple:
        scores_train, sentence_pairs_train, r1 = load_data(language=self.language, dataset='_train')
        scores_dev, sentence_pairs_dev, r2 = load_data(language=self.language, dataset='_dev_with_labels')
        scores_test, sentence_pairs_test, r3 = load_data(language=self.language, dataset='_test_with_labels')
        return scores_train, scores_dev, scores_test, \
            sentence_pairs_train, sentence_pairs_dev, sentence_pairs_test, \
            r1 or r2 or r3

    def calculate_spearman_correlation(self, true_scores, predicted_scores):
        self.spearman_correlation, _ = spearmanr(true_scores, predicted_scores)

    def print_results(self, model_name: str, dataset: str = 'Test') -> None:
        print(f'Model:                {model_name}')
        print(f'Language:             {FULL[self.language]}')
        print(f'Set:                  {dataset}')
        print(f'Spearman Correlation: {self.spearman_correlation:.3f}')
        if self.warning:
            print(f'WARNING: Some datasets were missing and were replaced with existing datasets')
        print()
