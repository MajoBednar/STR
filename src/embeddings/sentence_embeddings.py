import numpy as np


def sentence_pairs_to_pair_of_sentences(sentence_pairs: list[list[str]]) -> tuple[list[str], list[str]]:
    list_1, list_2 = zip(*sentence_pairs)
    return list(list_1), list(list_2)


def create_sentence_embeddings(model, sentence_pairs: list[list[str]]) -> tuple:
    pair_of_sentences = sentence_pairs_to_pair_of_sentences(sentence_pairs)
    sentence_embeddings1 = model.encode(pair_of_sentences[0])
    sentence_embeddings2 = model.encode(pair_of_sentences[1])
    return sentence_embeddings1, sentence_embeddings2


def sum_embeddings(embeddings1, embeddings2):
    return embeddings1 + embeddings2


def concat_embeddings(embeddings1, embeddings2):
    return np.concatenate((embeddings1, embeddings2), axis=1)
