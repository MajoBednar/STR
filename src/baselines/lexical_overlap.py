from nltk import ngrams
from scipy.stats import spearmanr

from src.utilities.program_args import parse_program_args
from src.utilities.load_data import load_data
from src.utilities.output_results import print_results


def calculate_dice_coefficient(sentence1: str, sentence2: str) -> float:
    n = 1
    ngram1 = ngrams(sentence1.split(), n)
    ngram2 = ngrams(sentence2.split(), n)
    ngram1 = set(ngram1)
    ngram2 = set(ngram2)
    overlap = 2 * len(ngram1.intersection(ngram2))
    return overlap / (len(ngram1) + len(ngram2))


def evaluate_lexical_overlap(language: str = 'eng') -> None:
    scores, sentence_pairs = load_data(language=language, dataset='_test_with_labels.csv')

    lexical_overlap_scores = []
    for pair in sentence_pairs:
        lexical_overlap_scores.append(calculate_dice_coefficient(pair[0], pair[1]))

    spearman_correlation, _ = spearmanr(scores, lexical_overlap_scores)
    model_name = 'Lexical Overlap'
    print_results(model_name, language, spearman_correlation)


if __name__ == "__main__":
    evaluate_lexical_overlap(language=parse_program_args())
