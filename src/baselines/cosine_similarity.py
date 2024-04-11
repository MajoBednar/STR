from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr

from src.embeddings.sentence_embeddings import create_sentence_embeddings
from src.models.base_model import BaseSTRLanguageModel
from src.utilities.metrics import compute_cosine_similarities
from src.utilities.program_args import parse_program_args


class STRCosineSimilarity(BaseSTRLanguageModel):
    def __init__(self, language: str):
        super().__init__(language)
        self.name = 'Cosine Similarity'
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

    def evaluate(self):
        embeddings1, embeddings2 = create_sentence_embeddings(self.sentence_transformer, self.sentence_pairs_test)
        similarity_scores = compute_cosine_similarities(embeddings1, embeddings2)

        self.spearman_correlation, _ = spearmanr(self.scores_test, similarity_scores)
        self.print_results()


if __name__ == '__main__':
    cosine_similarity = STRCosineSimilarity(language=parse_program_args())
    cosine_similarity.evaluate()
