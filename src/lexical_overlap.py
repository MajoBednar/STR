from nltk import ngrams
from maps import language_sentence_separator
import pandas as pd
from scipy.stats import spearmanr


def calculate_dice_coefficient(sentence1: str, sentence2: str) -> float:
    n = 1
    ngram1 = ngrams(sentence1.split(), n)
    ngram2 = ngrams(sentence2.split(), n)
    ngram1 = set(ngram1)
    ngram2 = set(ngram2)
    overlap = 2 * len(ngram1.intersection(ngram2))
    return overlap / (len(ngram1) + len(ngram2))


def evaluate():
    language = input('Language: ')
    test_data_path = '../data/' + language + '/' + language + '_test_with_labels.csv'
    test_df = pd.read_csv(test_data_path)

    scores = test_df["Score"].tolist()
    scores = [float(score) for score in scores]

    sentences = test_df["Text"].tolist()
    sentence_1s = []
    sentence_2s = []
    for sentence in sentences:
        sentence_1, sentence_2 = sentence.split(
            language_sentence_separator[language]
        )
        sentence_1s.append(sentence_1)
        sentence_2s.append(sentence_2)

    lexical_overlap_scores = []
    for sent1, sent2 in zip(sentence_1s, sentence_2s):
        lexical_overlap_scores.append(calculate_dice_coefficient(sent1, sent2))

    spearman_correlation, _ = spearmanr(scores, lexical_overlap_scores)
    print(spearman_correlation)


if __name__ == "__main__":
    evaluate()
