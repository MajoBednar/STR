import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from src.utilities.program_args import parse_program_args
from src.utilities.load_data import load_data, sentence_pairs_to_pair_of_sentences
from src.utilities.metrics import cosine_similarities


def train(language: str = 'eng'):
    scores, sentence_pairs = load_data(language=language, dataset='_dev_with_labels.csv')
    pair_of_sentences = sentence_pairs_to_pair_of_sentences(sentence_pairs)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings1 = model.encode(pair_of_sentences[0])
    embeddings2 = model.encode(pair_of_sentences[1])

    regressor = LinearRegression()
    regressor.fit(np.concatenate((embeddings1, embeddings2), axis=1), scores)

    prediction_scores = regressor.predict(np.concatenate((embeddings1, embeddings2), axis=1))
    spearman_correlation, _ = spearmanr(scores, prediction_scores)
    print(f'Correlation between lr and train scores:            {spearman_correlation}')

    return regressor


def main(language: str = 'eng') -> None:
    scores, sentence_pairs = load_data(language=language, dataset='_test_with_labels.csv')
    pair_of_sentences = sentence_pairs_to_pair_of_sentences(sentence_pairs)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings1 = model.encode(pair_of_sentences[0])
    embeddings2 = model.encode(pair_of_sentences[1])

    similarity_scores = cosine_similarities(embeddings1, embeddings2)
    spearman_correlation, _ = spearmanr(scores, similarity_scores)
    print(f'Correlation between cosine similarities and scores: {spearman_correlation}')

    regressor = train(language)
    prediction_scores = regressor.predict(np.concatenate((embeddings1, embeddings2), axis=1))
    spearman_correlation, _ = spearmanr(scores, prediction_scores)
    print(f'Correlation between linear regression and scores:   {spearman_correlation}')


if __name__ == '__main__':
    main(parse_program_args())
