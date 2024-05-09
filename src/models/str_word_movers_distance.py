import numpy as np
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity

from src.embeddings.token_embeddings import DataManagerWithTokenEmbeddings
from src.utilities.program_args import parse_program_args


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
    cosine_similarity = STRCosineSimilarity(language=language, data_split=data_split, transformer_name='LaBSE')
    cosine_similarity.evaluate('Train')
    cosine_similarity.evaluate('Dev')
    cosine_similarity.evaluate()


if __name__ == '__main__':
    main()
