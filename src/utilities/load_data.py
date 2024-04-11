import pandas as pd

from .constants import SENTENCE_SEPARATOR_FOR_LANGAUGE as SEPARATOR


def load_scores(df: pd.DataFrame) -> list[float]:
    scores = df["Score"].tolist()
    scores = [float(score) for score in scores]
    return scores


def load_sentence_pairs(df: pd.DataFrame, language: str) -> list[list[str]]:
    sentences = df["Text"].tolist()
    sentence_pairs = []
    for sentence in sentences:
        sentence_1, sentence_2 = sentence.split(SEPARATOR[language])
        sentence_pairs.append([sentence_1, sentence_2])
    return sentence_pairs


def load_data(language: str, dataset: str) -> tuple[list[float], list[list[str]]]:
    data_path = 'data/datasets_original_splits/' + language + '/' + language + dataset + '.csv'
    df = pd.read_csv(data_path)
    scores = load_scores(df)
    sentence_pairs = load_sentence_pairs(df, language)
    return scores, sentence_pairs
