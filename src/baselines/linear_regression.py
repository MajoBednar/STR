from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression

from src.utilities.program_args import parse_program_args
from src.utilities.data_management import DataManager
from src.embeddings.sentence_embeddings import create_sentence_embeddings, sum_embeddings, concat_embeddings


class STRLinearRegression:
    def __init__(self, language: str, pooling_function):
        self.name = 'Linear Regression by '
        self.name += 'Summing' if pooling_function == sum_embeddings else 'Concatenating'
        self.name += ' Sentence Embeddings'

        self.data = DataManager(language)
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.regressor = LinearRegression()
        self.pooling_function = pooling_function

    def train(self) -> None:
        embeddings1, embeddings2 = create_sentence_embeddings(self.sentence_transformer, self.data.sentence_pairs_train
                                                              + self.data.sentence_pairs_dev)
        pooled_embeddings = self.pooling_function(embeddings1, embeddings2)
        self.regressor.fit(pooled_embeddings, self.data.scores_train + self.data.scores_dev)

        predicted_scores = self.regressor.predict(pooled_embeddings)
        self.data.calculate_spearman_correlation(self.data.scores_train + self.data.scores_dev, predicted_scores)
        self.data.print_results(self.name, 'Training + Development')

    def evaluate(self) -> None:
        embeddings1, embeddings2 = create_sentence_embeddings(self.sentence_transformer, self.data.sentence_pairs_test)
        pooled_embeddings = self.pooling_function(embeddings1, embeddings2)
        predicted_scores = self.regressor.predict(pooled_embeddings)

        self.data.calculate_spearman_correlation(self.data.scores_test, predicted_scores)
        self.data.print_results(self.name)


if __name__ == '__main__':
    linear_regression_sum = STRLinearRegression(language=parse_program_args(), pooling_function=sum_embeddings)
    linear_regression_sum.train()
    linear_regression_sum.evaluate()

    linear_regression_concat = STRLinearRegression(language=parse_program_args(), pooling_function=concat_embeddings)
    linear_regression_concat.train()
    linear_regression_concat.evaluate()
