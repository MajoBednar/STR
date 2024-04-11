from sentence_transformers import SentenceTransformer

from src.embeddings.sentence_embeddings import create_sentence_embeddings
from src.utilities.data_management import DataManager
from src.utilities.metrics import compute_cosine_similarities
from src.utilities.program_args import parse_program_args


class STRCosineSimilarity:
    def __init__(self, language: str):
        self.name = 'Cosine Similarity'
        self.data = DataManager(language)
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

    def evaluate(self):
        embeddings1, embeddings2 = create_sentence_embeddings(self.sentence_transformer, self.data.sentence_pairs_test)
        similarity_scores = compute_cosine_similarities(embeddings1, embeddings2)

        self.data.calculate_spearman_correlation(self.data.scores_test, similarity_scores)
        self.data.print_results(self.name)


if __name__ == '__main__':
    cosine_similarity = STRCosineSimilarity(language=parse_program_args())
    cosine_similarity.evaluate()
