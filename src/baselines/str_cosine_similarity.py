import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import paired_cosine_distances

from src.embeddings.sentence_embeddings import DataManagerWithSentenceEmbeddings
from src.utilities.program_args import parse_program_args


def compute_cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = norm(vector1)
    norm_vector2 = norm(vector2)
    return dot_product / (norm_vector1 * norm_vector2)


def compute_cosine_similarities(vectors1: object, vectors2: object) -> list[float]:
    cosine_scores = 1 - paired_cosine_distances(vectors1, vectors2)
    cosine_scores = cosine_scores.flatten().tolist()
    return cosine_scores


class STRCosineSimilarity:
    def __init__(self, data_manager: DataManagerWithSentenceEmbeddings):
        self.name: str = 'Cosine Similarity'
        self.data: DataManagerWithSentenceEmbeddings = data_manager

    def evaluate(self, dataset: str = 'Test') -> float:
        similarity_scores = compute_cosine_similarities(self.data.sentence_embeddings[dataset][0],
                                                        self.data.sentence_embeddings[dataset][1])
        correlation = self.data.calculate_spearman_correlation(self.data.scores[dataset], similarity_scores)
        self.data.print_results(correlation, self.name, self.data.transformer_name, dataset)
        return correlation


def evaluate_cosine_similarity(data_manager: DataManagerWithSentenceEmbeddings) -> None:
    cosine_similarity = STRCosineSimilarity(data_manager)
    cosine_similarity.evaluate('Test')


def main() -> None:
    language, data_split = parse_program_args()
    data_manager = DataManagerWithSentenceEmbeddings.load(language, data_split, 'LaBSE')

    cosine_similarity = STRCosineSimilarity(data_manager)
    cosine_similarity.evaluate('Train')
    cosine_similarity.evaluate('Dev')
    cosine_similarity.evaluate()


if __name__ == '__main__':
    main()
