from nltk import ngrams
from scipy.stats import spearmanr

from src.models.base_model import BaseSTRLanguageModel
from src.utilities.program_args import parse_program_args


class STRLexicalOverlap(BaseSTRLanguageModel):
    def __init__(self, language: str):
        super().__init__(language)
        self.name = 'Lexical Overlap'

    @staticmethod
    def calculate_dice_coefficient(sentence1: str, sentence2: str) -> float:
        n = 1
        ngram1 = ngrams(sentence1.split(), n)
        ngram2 = ngrams(sentence2.split(), n)
        ngram1 = set(ngram1)
        ngram2 = set(ngram2)
        overlap = 2 * len(ngram1.intersection(ngram2))
        return overlap / (len(ngram1) + len(ngram2))

    def evaluate(self) -> None:
        lexical_overlap_scores = []
        for pair in self.sentence_pairs_test:
            lexical_overlap_scores.append(self.calculate_dice_coefficient(pair[0], pair[1]))

        self.spearman_correlation, _ = spearmanr(self.scores_test, lexical_overlap_scores)
        self.print_results()


if __name__ == "__main__":
    lexical_overlap = STRLexicalOverlap(language=parse_program_args())
    lexical_overlap.evaluate()
