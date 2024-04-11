from src.embeddings.sentence_embeddings import DataManagerWithSentenceEmbeddings
from src.utilities.metrics import compute_cosine_similarities
from src.utilities.program_args import parse_program_args


class STRCosineSimilarity:
    def __init__(self, language: str):
        self.name = 'Cosine Similarity'
        self.data = DataManagerWithSentenceEmbeddings(language)

    def evaluate(self, dataset: str = 'Test') -> None:
        similarity_scores = compute_cosine_similarities(self.data.sentence_embeddings[dataset][0],
                                                        self.data.sentence_embeddings[dataset][1])

        self.data.calculate_spearman_correlation(self.data.scores[dataset], similarity_scores)
        self.data.print_results(self.name, dataset)


if __name__ == '__main__':
    cosine_similarity = STRCosineSimilarity(language=parse_program_args())
    cosine_similarity.evaluate('Train')
    cosine_similarity.evaluate('Dev')
    cosine_similarity.evaluate()
