from src.utilities.load_data import load_data
from src.utilities.output_results import print_results


class BaseSTRLanguageModel:
    def __init__(self, language):
        self.name: str = 'NAME NOT PROVIDED'
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

    def evaluate(self):
        pass

    def print_results(self):
        print_results(model_name=self.name, language=self.language, spearman_correlation=self.spearman_correlation)
