import pandas as pd
import os
import numpy as np
from scipy.stats import spearmanr

from .constants import SENTENCE_SEPARATOR as SEP, FULL_LANGUAGE_NAME as FULL


class DataManager:
    def __init__(self, language):
        self.language: str = language
        self.warning = False

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

    def set_spearman_correlation(self, true_scores, predicted_scores):
        self.spearman_correlation = self.calculate_spearman_correlation(true_scores, predicted_scores)

    @staticmethod
    def calculate_spearman_correlation(true_scores, predicted_scores):
        spearman_correlation, _ = spearmanr(true_scores, predicted_scores)
        return spearman_correlation

    def print_results(self, relatedness_model: str, transformer_model: str, dataset: str = 'Test') -> None:
        print(f'Language:             {FULL[self.language]}')
        print(f'Transformer Model:    {transformer_model}')
        print(f'STR Model:            {relatedness_model}')
        print(f'Set:                  {dataset}')
        print(f'Spearman Correlation: {self.spearman_correlation:.3f}')
        if self.warning:
            print(f'WARNING: Some datasets were missing and were replaced with existing datasets')
        print()

    @staticmethod
    def sentence_pairs_to_pair_of_sentences(sentence_pairs: list[list[str]]) -> tuple[list[str], list[str]]:
        list_1, list_2 = zip(*sentence_pairs)
        return list(list_1), list(list_2)

    @staticmethod
    def _embeddings_train_dev(train_embeddings, dev_embeddings) -> tuple:
        train_dev1 = np.concatenate((train_embeddings[0], dev_embeddings[0]), axis=0)
        train_dev2 = np.concatenate((train_embeddings[1], dev_embeddings[1]), axis=0)
        return train_dev1, train_dev2

    def __initialize_data(self) -> tuple:
        scores_train, sentence_pairs_train = self.__load_data(dataset='_train')
        scores_dev, sentence_pairs_dev = self.__load_data(dataset='_dev_with_labels')
        scores_test, sentence_pairs_test = self.__load_data(dataset='_test_with_labels')
        return scores_train, scores_dev, scores_test, sentence_pairs_train, sentence_pairs_dev, sentence_pairs_test

    def __load_data(self, dataset: str) -> tuple[list[float], list[list[str]]]:
        path = 'data/datasets_original_splits/' + self.language + '/'
        file = self.language + dataset + '.csv'
        if not os.path.exists(path + file):
            file = self.__replace_missing_dataset(dataset=dataset)
            self.warning = True
        df = pd.read_csv(path + file)
        scores = self.__load_scores(df)
        sentence_pairs = self.__load_sentence_pairs(df)
        return scores, sentence_pairs

    @staticmethod
    def __load_scores(df: pd.DataFrame) -> list[float]:
        scores = df['Score'].tolist()
        scores = [float(score) for score in scores]
        return scores

    @staticmethod
    def __load_sentence_pairs(df: pd.DataFrame) -> list[list[str]]:
        sentences = df["Text"].tolist()
        sentence_pairs = []
        for sentence in sentences:
            sentence_1, sentence_2 = sentence.split(SEP)
            sentence_pairs.append([sentence_1, sentence_2])
        return sentence_pairs

    def __replace_missing_dataset(self, dataset: str) -> str:
        new_dataset = 'REPLACEMENT FAILED'
        match dataset:
            case '_train' | '_test_with_labels':
                new_dataset = '_dev_with_labels'
        self.__print_missing_dataset_warning(dataset, new_dataset)
        return self.language + new_dataset + '.csv'

    def __print_missing_dataset_warning(self, old_dataset: str, new_dataset: str) -> None:
        print(f'WARNING: {self.language + old_dataset} dataset is missing and being replaced with '
              f'{self.language + new_dataset} dataset')
