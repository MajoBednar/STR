import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from src.utilities.program_args import parse_program_args
from src.models.base_model import BaseSTRLanguageModel
from src.embeddings.sentence_embeddings import create_sentence_embeddings
from src.utilities.output_results import print_results


class STRLinearRegression(BaseSTRLanguageModel):
    def __init__(self, language: str):
        super().__init__(language)
        self.name = 'Linear Regression by Concatenating Sentence Embeddings'
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.regressor_concat = LinearRegression()

    def train(self) -> None:
        embeddings1, embeddings2 = create_sentence_embeddings(self.sentence_transformer,
                                                              self.sentence_pairs_train + self.sentence_pairs_dev)
        concatenated_embeddings = np.concatenate((embeddings1, embeddings2), axis=1)
        self.regressor_concat.fit(concatenated_embeddings, self.scores_train + self.scores_dev)

        prediction_scores_concat = self.regressor_concat.predict(concatenated_embeddings)
        spearman_correlation_concat, _ = spearmanr(self.scores_train + self.scores_dev, prediction_scores_concat)
        print_results(self.name + ' by Concatenating TRAINING', self.language, spearman_correlation_concat)

    def evaluate(self) -> None:
        embeddings1, embeddings2 = create_sentence_embeddings(self.sentence_transformer, self.sentence_pairs_test)
        concatenated_embeddings = np.concatenate((embeddings1, embeddings2), axis=1)
        prediction_scores_concat = self.regressor_concat.predict(concatenated_embeddings)

        self.spearman_correlation, _ = spearmanr(self.scores_test, prediction_scores_concat)
        self.print_results()


if __name__ == '__main__':
    linear_regression = STRLinearRegression(language=parse_program_args())
    linear_regression.train()
    linear_regression.evaluate()
