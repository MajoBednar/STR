from sklearn.linear_model import LinearRegression

from src.utilities.program_args import parse_program_args
from src.embeddings.sentence_embeddings import sum_embeddings, concat_embeddings, DataManagerWithSentenceEmbeddings


class STRLinearRegression:
    def __init__(self, language: str, pooling_function):
        self.name = 'Linear Regression by '
        self.name += 'Summing' if pooling_function == sum_embeddings else 'Concatenating'
        self.name += ' Sentence Embeddings'

        self.data = DataManagerWithSentenceEmbeddings.load(language)
        self.regressor = LinearRegression()
        self.pooling_function = pooling_function

    def train(self) -> None:
        dataset = 'Train+Dev'
        embeddings1, embeddings2 = self.data.sentence_embeddings[dataset]
        pooled_embeddings = self.pooling_function(embeddings1, embeddings2)
        self.regressor.fit(pooled_embeddings, self.data.scores[dataset])

        predicted_scores = self.regressor.predict(pooled_embeddings)
        self.data.calculate_spearman_correlation(self.data.scores[dataset], predicted_scores)
        self.data.print_results(self.name, dataset)

    def evaluate(self, dataset: str = 'Test') -> None:
        embeddings1, embeddings2 = self.data.sentence_embeddings[dataset]
        pooled_embeddings = self.pooling_function(embeddings1, embeddings2)
        predicted_scores = self.regressor.predict(pooled_embeddings)

        self.data.calculate_spearman_correlation(self.data.scores[dataset], predicted_scores)
        self.data.print_results(self.name, dataset)


if __name__ == '__main__':
    linear_regression_sum = STRLinearRegression(language=parse_program_args(), pooling_function=sum_embeddings)
    linear_regression_sum.train()
    linear_regression_sum.evaluate()

    linear_regression_concat = STRLinearRegression(language=parse_program_args(), pooling_function=concat_embeddings)
    linear_regression_concat.train()
    linear_regression_concat.evaluate()
