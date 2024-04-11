import numpy as np
from sentence_transformers import SentenceTransformer

from src.utilities.data_management import DataManager


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


class DataManagerWithSentenceEmbeddings(DataManager):
    def __init__(self, language, sentence_transformer_model: str = 'all-MiniLM-L6-v2'):
        super().__init__(language)
        self.sentence_transformer = SentenceTransformer(sentence_transformer_model)

        self.sentence_embeddings_train = create_sentence_embeddings(self.sentence_transformer,
                                                                    self.sentence_pairs_train)
        self.sentence_embeddings_dev = create_sentence_embeddings(self.sentence_transformer, self.sentence_pairs_dev)
        self.sentence_embeddings_test = create_sentence_embeddings(self.sentence_transformer, self.sentence_pairs_test)

        self.embedding_dim = len(self.sentence_embeddings_train[0][0])

    def sentence_embeddings_train_dev(self) -> tuple:
        train_dev1 = np.concatenate((self.sentence_embeddings_train[0], self.sentence_embeddings_dev[0]), axis=0)
        train_dev2 = np.concatenate((self.sentence_embeddings_train[1], self.sentence_embeddings_dev[1]), axis=0)
        return train_dev1, train_dev2
