from maps import language_sentence_separator
import pandas as pd


def load_scores(df: pd.DataFrame) -> list[float]:
    scores = df["Score"].tolist()
    scores = [float(score) for score in scores]
    return scores


def load_sentence_pairs(df: pd.DataFrame, language: str = 'eng') -> list[list[str]]:
    sentences = df["Text"].tolist()
    sentence_pairs = []
    for sentence in sentences:
        sentence_1, sentence_2 = sentence.split(language_sentence_separator[language])
        sentence_pairs.append([sentence_1, sentence_2])
    return sentence_pairs


def load_data(language: str = 'eng', dataset: str = '_train') -> (list[float], list[list[str]]):
    data_path = '../data/' + language + '/' + language + dataset
    df = pd.read_csv(data_path)
    scores = load_scores(df)
    sentence_pairs = load_sentence_pairs(df, language)
    return scores, sentence_pairs
