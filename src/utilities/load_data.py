import pandas as pd

from .constants import SENTENCE_SEPARATOR_FOR_LANGAUGE as SEPARATOR


def load_scores(df: pd.DataFrame) -> list[float]:
    scores = df["Score"].tolist()
    scores = [float(score) for score in scores]
    return scores


def load_sentence_pairs(df: pd.DataFrame, language: str = 'eng') -> list[list[str]]:
    sentences = df["Text"].tolist()
    sentence_pairs = []
    for sentence in sentences:
        sentence_1, sentence_2 = sentence.split(SEPARATOR[language])
        sentence_pairs.append([sentence_1, sentence_2])
    return sentence_pairs


def load_data(language: str = 'eng', dataset: str = '_train') -> tuple[list[float], list[list[str]]]:
    data_path = 'data/datasets_original_splits/' + language + '/' + language + dataset
    df = pd.read_csv(data_path)
    scores = load_scores(df)
    sentence_pairs = load_sentence_pairs(df, language)
    return scores, sentence_pairs


def sentence_pairs_to_pair_of_sentences(sentence_pairs: list[list[str]]) -> tuple[list[str], list[str]]:
    list_1, list_2 = zip(*sentence_pairs)
    return list(list_1), list(list_2)
