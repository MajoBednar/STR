from sklearn.linear_model import LinearRegression

from src.utilities.program_args import parse_program_args
from src.utilities.constants import Verbose
from src.embeddings.sentence_embeddings import sum_embeddings, concat_embeddings, DataManagerWithSentenceEmbeddings


class STRLinearRegression:
    def __init__(self, language: str, pooling_function, verbose: Verbose = Verbose.DEFAULT):
        self.name = 'Linear Regression by '
        self.name += 'Summing' if pooling_function == sum_embeddings else 'Concatenating'
        self.name += ' Sentence Embeddings'
        self.verbose: Verbose = verbose

        self.data = DataManagerWithSentenceEmbeddings.load(language)
        self.regressor = LinearRegression()
        self.pooling_function = pooling_function

    def train(self, dataset: str = 'Train+Dev') -> None:
        embeddings1, embeddings2 = self.data.sentence_embeddings[dataset]
        pooled_embeddings = self.pooling_function(embeddings1, embeddings2)
        self.regressor.fit(pooled_embeddings, self.data.scores[dataset])

        predicted_scores = self.regressor.predict(pooled_embeddings)
        self.data.set_spearman_correlation(self.data.scores[dataset], predicted_scores)
        if self.verbose == Verbose.DEFAULT or self.verbose == Verbose.EXPRESSIVE:
            self.data.print_results(self.name, dataset)

    def evaluate(self, dataset: str = 'Test') -> None:
        embeddings1, embeddings2 = self.data.sentence_embeddings[dataset]
        pooled_embeddings = self.pooling_function(embeddings1, embeddings2)
        predicted_scores = self.regressor.predict(pooled_embeddings)

        self.data.set_spearman_correlation(self.data.scores[dataset], predicted_scores)
        self.data.print_results(self.name, dataset)


def evaluate_linear_regression(language: str) -> None:
    lr_sum = STRLinearRegression(language=language, pooling_function=sum_embeddings, verbose=Verbose.SILENT)
    lr_sum.train(dataset='Train')
    lr_sum.evaluate()

    lr_concat = STRLinearRegression(language=language, pooling_function=concat_embeddings, verbose=Verbose.SILENT)
    lr_concat.train(dataset='Train')
    lr_concat.evaluate()


def main() -> None:
    lr_sum = STRLinearRegression(language=parse_program_args(), pooling_function=sum_embeddings)
    lr_sum.train()
    lr_sum.evaluate()

    lr_concat = STRLinearRegression(language=parse_program_args(), pooling_function=concat_embeddings)
    lr_concat.train()
    lr_concat.evaluate()


if __name__ == '__main__':
    main()
