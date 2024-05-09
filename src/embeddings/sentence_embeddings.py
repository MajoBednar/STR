from sentence_transformers import SentenceTransformer
import pickle as pkl
import os

from src.utilities.data_management import DataManager
from src.utilities.constants import SENTENCE_TRANSFORMERS as ST


class DataManagerWithSentenceEmbeddings(DataManager):
    def __init__(self, language: str, data_split: str, sentence_transformer_model_name: str, save_data: bool):
        super().__init__(language, data_split)
        self.transformer_name = sentence_transformer_model_name + ' Sentence Transformer'
        self.sentence_transformer = SentenceTransformer(ST[sentence_transformer_model_name])

        self.sentence_embeddings = {
            'Train': self.__create_sentence_embeddings(self.sentence_pairs['Train']),
            'Dev': self.__create_sentence_embeddings(self.sentence_pairs['Dev']),
            'Test': self.__create_sentence_embeddings(self.sentence_pairs['Test'])
        }
        self.embedding_dim = len(self.sentence_embeddings['Train'][0][0])

        if save_data is True:
            self.sentence_transformer = None
            self._save(sentence_transformer_model_name)

    def get_embeddings(self):
        return self.sentence_embeddings

    def __create_sentence_embeddings(self, sentence_pairs: list[list[str]]) -> tuple:
        pair_of_sentences = DataManager.sentence_pairs_to_pair_of_sentences(sentence_pairs)
        sentence_embeddings1 = self.sentence_transformer.encode(pair_of_sentences[0], convert_to_tensor=True)
        sentence_embeddings2 = self.sentence_transformer.encode(pair_of_sentences[1], convert_to_tensor=True)
        return sentence_embeddings1, sentence_embeddings2

    def _save(self, transformer_model: str, directory: str = 'data/sentence_embeddings/'):
        super()._save(transformer_model, directory)

    @staticmethod
    def load(language: str, data_split: str, sentence_transformer_model: str, save_data: bool = True):
        path = 'data/sentence_embeddings/' + sentence_transformer_model + '_' + language + '_' + data_split + '.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as file:
                return pkl.load(file)
        return DataManagerWithSentenceEmbeddings(language, data_split, sentence_transformer_model, save_data)
