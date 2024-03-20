from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr

from program_args import parse_program_args
from load_data import load_data, sentence_pairs_to_pair_of_sentences
from metrics import cosine_similarities


def main(language: str = 'eng') -> None:
    scores, sentence_pairs = load_data(language=language, dataset='_test_with_labels.csv')
    pair_of_sentences = sentence_pairs_to_pair_of_sentences(sentence_pairs)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings1 = model.encode(pair_of_sentences[0])
    embeddings2 = model.encode(pair_of_sentences[1])

    similarity_scores = cosine_similarities(embeddings1, embeddings2)
    spearman_correlation, _ = spearmanr(scores, similarity_scores)
    print(spearman_correlation)


if __name__ == '__main__':
    main(parse_program_args())
