from src.embeddings.sentence_embeddings import DataManagerWithSentenceEmbeddings
from src.utilities.metrics import compute_cosine_similarities
from src.utilities.program_args import parse_program_args


class STRCosineSimilarity:
    def __init__(self, language: str):
        self.name = 'Cosine Similarity'
        self.data = DataManagerWithSentenceEmbeddings(language)

    def evaluate(self):
        similarity_scores = compute_cosine_similarities(self.data.sentence_embeddings_test[0],
                                                        self.data.sentence_embeddings_test[1])

        self.data.calculate_spearman_correlation(self.data.scores_test, similarity_scores)
        self.data.print_results(self.name)


if __name__ == '__main__':
    cosine_similarity = STRCosineSimilarity(language=parse_program_args())
    cosine_similarity.evaluate()
