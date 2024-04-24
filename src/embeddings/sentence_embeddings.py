import numpy as np
from sentence_transformers import SentenceTransformer
import pickle as pkl
import os

from src.utilities.data_management import DataManager


def sentence_pairs_to_pair_of_sentences(sentence_pairs: list[list[str]]) -> tuple[list[str], list[str]]:
    list_1, list_2 = zip(*sentence_pairs)
    return list(list_1), list(list_2)


def sum_embeddings(embeddings1, embeddings2):
    return embeddings1 + embeddings2


def concat_embeddings(embeddings1, embeddings2):
    return np.concatenate((embeddings1, embeddings2), axis=1)


class DataManagerWithSentenceEmbeddings(DataManager):
    def __init__(self, language, sentence_transformer_model: str = 'all-MiniLM-L6-v2'):
        super().__init__(language)
        self.sentence_transformer = SentenceTransformer(sentence_transformer_model)

        self.sentence_embeddings = {
            'Train': self.__create_sentence_embeddings(self.sentence_pairs['Train']),
            'Dev': self.__create_sentence_embeddings(self.sentence_pairs['Dev']),
            'Test': self.__create_sentence_embeddings(self.sentence_pairs['Test']),
        }
        self.__sentence_embeddings_train_dev()
        self.embedding_dim = len(self.sentence_embeddings['Train'][0][0])

        self._save(sentence_transformer_model)

    def __create_sentence_embeddings(self, sentence_pairs: list[list[str]]) -> tuple:
        pair_of_sentences = sentence_pairs_to_pair_of_sentences(sentence_pairs)
        sentence_embeddings1 = self.sentence_transformer.encode(pair_of_sentences[0])
        sentence_embeddings2 = self.sentence_transformer.encode(pair_of_sentences[1])
        return sentence_embeddings1, sentence_embeddings2

    def __sentence_embeddings_train_dev(self) -> None:
        train_dev1 = np.concatenate((self.sentence_embeddings['Train'][0], self.sentence_embeddings['Dev'][0]), axis=0)
        train_dev2 = np.concatenate((self.sentence_embeddings['Train'][1], self.sentence_embeddings['Dev'][1]), axis=0)
        self.sentence_embeddings['Train+Dev'] = train_dev1, train_dev2

    def _save(self, sentence_transformer_model: str):
        directory = 'data/sentence_embeddings/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        path = directory + sentence_transformer_model + '_' + self.language + '.pkl'
        with open(path, 'wb') as file:
            pkl.dump(self, file)

    @staticmethod
    def load(language: str, sentence_transformer_model: str = 'all-MiniLM-L6-v2'):
        path = 'data/sentence_embeddings/' + sentence_transformer_model + '_' + language + '.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as file:
                return pkl.load(file)
        return DataManagerWithSentenceEmbeddings(language)
