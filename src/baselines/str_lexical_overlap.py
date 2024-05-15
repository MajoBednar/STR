from nltk import ngrams

from src.utilities.data_management import DataManager
from src.utilities.program_args import parse_program_args


class STRLexicalOverlap:
    def __init__(self, data_manager: DataManager):
        self.name: str = 'Lexical Overlap'
        self.data: DataManager = data_manager

    @staticmethod
    def calculate_dice_coefficient(sentence1: str, sentence2: str, n: int = 1) -> float:
        # computes score in [0, 1] based on lexical overlap between two sentences
        ngram1 = ngrams(sentence1.split(), n)
        ngram2 = ngrams(sentence2.split(), n)
        ngram1 = set(ngram1)
        ngram2 = set(ngram2)
        overlap = 2 * len(ngram1.intersection(ngram2))
        return overlap / (len(ngram1) + len(ngram2))

    def evaluate(self, dataset: str = 'Test') -> float:
        lexical_overlap_scores = []
        for pair in self.data.sentence_pairs[dataset]:
            lexical_overlap_scores.append(self.calculate_dice_coefficient(pair[0], pair[1]))

        correlation = self.data.calculate_spearman_correlation(self.data.scores[dataset], lexical_overlap_scores)
        self.data.print_results(correlation, relatedness_model=self.name, dataset=dataset)
        return correlation


def evaluate_lexical_overlap(data_manager: DataManager) -> None:
    lexical_overlap = STRLexicalOverlap(data_manager)
    lexical_overlap.evaluate('Test')


def main() -> None:
    language, data_split = parse_program_args()
    data_manager = DataManager(language, data_split)

    lexical_overlap = STRLexicalOverlap(data_manager)
    lexical_overlap.evaluate('Train')
    lexical_overlap.evaluate('Dev')
    lexical_overlap.evaluate('Test')


if __name__ == '__main__':
    main()
