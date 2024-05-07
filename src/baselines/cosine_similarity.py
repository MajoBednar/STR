import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import paired_cosine_distances

from src.embeddings.sentence_embeddings import DataManagerWithSentenceEmbeddings
from src.utilities.program_args import parse_program_args


def compute_cosine_similarity(vector1, vector2) -> float:
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = norm(vector1)
    norm_vector2 = norm(vector2)
    return dot_product / (norm_vector1 * norm_vector2)


def compute_cosine_similarities(vectors1, vectors2) -> list[float]:
    cosine_scores = 1 - paired_cosine_distances(vectors1, vectors2)
    cosine_scores = cosine_scores.flatten().tolist()
    return cosine_scores


class STRCosineSimilarity:
    def __init__(self, language: str, data_split: str, transformer_name: str):
        self.name = 'Cosine Similarity'
        self.data = DataManagerWithSentenceEmbeddings.load(language, data_split, transformer_name)

    def evaluate(self, dataset: str = 'Test') -> None:
        similarity_scores = compute_cosine_similarities(self.data.sentence_embeddings[dataset][0],
                                                        self.data.sentence_embeddings[dataset][1])

        self.data.set_spearman_correlation(self.data.scores[dataset], similarity_scores)
        self.data.print_results(self.name, self.data.transformer_name, dataset)


def evaluate_cosine_similarity(language: str, data_split: str, transformer_name: str) -> None:
    cosine_similarity = STRCosineSimilarity(language=language, data_split=data_split, transformer_name=transformer_name)
    cosine_similarity.evaluate('Test')


def main() -> None:
    language, data_split = parse_program_args()
    cosine_similarity = STRCosineSimilarity(language=language, data_split=data_split, transformer_name='all MiniLM')
    cosine_similarity.evaluate('Train')
    cosine_similarity.evaluate('Dev')
    cosine_similarity.evaluate()


if __name__ == '__main__':
    main()
