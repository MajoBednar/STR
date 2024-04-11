from nltk import ngrams

from src.utilities.data_management import DataManager
from src.utilities.program_args import parse_program_args


class STRLexicalOverlap:
    def __init__(self, language: str):
        self.name = 'Lexical Overlap'
        self.data = DataManager(language)

    @staticmethod
    def calculate_dice_coefficient(sentence1: str, sentence2: str) -> float:
        n = 1
        ngram1 = ngrams(sentence1.split(), n)
        ngram2 = ngrams(sentence2.split(), n)
        ngram1 = set(ngram1)
        ngram2 = set(ngram2)
        overlap = 2 * len(ngram1.intersection(ngram2))
        return overlap / (len(ngram1) + len(ngram2))

    def evaluate(self, dataset: str = 'Test') -> None:
        lexical_overlap_scores = []
        for pair in self.data.sentence_pairs[dataset]:
            lexical_overlap_scores.append(self.calculate_dice_coefficient(pair[0], pair[1]))

        self.data.calculate_spearman_correlation(self.data.scores[dataset], lexical_overlap_scores)
        self.data.print_results(self.name, dataset)


if __name__ == "__main__":
    lexical_overlap = STRLexicalOverlap(language=parse_program_args())
    lexical_overlap.evaluate('Train')
    lexical_overlap.evaluate('Dev')
    lexical_overlap.evaluate('Test')
