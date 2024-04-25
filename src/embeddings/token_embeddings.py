from transformers import AutoTokenizer, AutoModel
import torch
import os
import pickle as pkl

from src.utilities.data_management import DataManager


class DataManagerWithTokenEmbeddings(DataManager):
    def __init__(self, language, token_transformer_model_name: str = 'bert-base-uncased'):
        super().__init__(language)
        self.tokenizer = AutoTokenizer.from_pretrained(token_transformer_model_name)
        self.token_transformer = AutoModel.from_pretrained(token_transformer_model_name)

        self.token_embeddings = {
            'Train': self.__create_token_embeddings(self.sentence_pairs['Train']),
            'Dev': self.__create_token_embeddings(self.sentence_pairs['Dev']),
            'Test': self.__create_token_embeddings(self.sentence_pairs['Test'])
        }
        self.__token_embeddings_train_dev()

        self.number_of_tokens = len(self.token_embeddings['Train'][0][0])  # number of tokens in each sentence
        self.embedding_dim = len(self.token_embeddings['Train'][0][0][0])

        self._save(token_transformer_model_name)

    def __create_token_embeddings(self, sentence_pairs: list[list[str]]) -> tuple:
        pair_of_sentences = DataManager.sentence_pairs_to_pair_of_sentences(sentence_pairs)
        tokenized_sentences1 = self.tokenizer(pair_of_sentences[0], return_tensors='pt', padding=True, truncation=True)
        tokenized_sentences2 = self.tokenizer(pair_of_sentences[1], return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            outputs1 = self.token_transformer(**tokenized_sentences1)
            outputs2 = self.token_transformer(**tokenized_sentences2)

        token_embeddings1 = outputs1.last_hidden_state
        token_embeddings2 = outputs2.last_hidden_state
        return token_embeddings1, token_embeddings2

    def __token_embeddings_train_dev(self) -> None:
        train_dev_embeddings = DataManager._embeddings_train_dev(self.token_embeddings['Train'],
                                                                 self.token_embeddings['Dev'])
        self.token_embeddings['Train+Dev'] = train_dev_embeddings

    def _save(self, token_transformer_model: str):
        directory = 'data/token_embeddings/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        path = directory + token_transformer_model + '_' + self.language + '.pkl'
        with open(path, 'wb') as file:
            pkl.dump(self, file)

    @staticmethod
    def load(language: str, token_transformer_model: str = 'bert-base-uncased'):
        path = 'data/token_embeddings/' + token_transformer_model + '_' + language + '.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as file:
                return pkl.load(file)
        return DataManagerWithTokenEmbeddings(language)
