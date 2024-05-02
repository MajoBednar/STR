from transformers import AutoTokenizer, AutoModel
import torch
import os
import pickle as pkl

from src.utilities.data_management import DataManager
from src.utilities.constants import TOKEN_TRANSFORMERS as TT


class DataManagerWithTokenEmbeddings(DataManager):
    def __init__(self, language: str, token_transformer_model_name: str, save_data: bool):
        super().__init__(language)
        self.token_transformer_name = token_transformer_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(TT[token_transformer_model_name])
        self.token_transformer = AutoModel.from_pretrained(TT[token_transformer_model_name])

        self.token_embeddings = {
            'Train': self.__create_token_embeddings(self.sentence_pairs['Train']),
            'Dev': self.__create_token_embeddings(self.sentence_pairs['Dev']),
            'Test': self.__create_token_embeddings(self.sentence_pairs['Test'])
        }
        # self.__token_embeddings_train_dev()

        self.number_of_tokens = len(self.token_embeddings['Train'][0][0])  # number of tokens in each sentence
        print('Number of tokens:', self.number_of_tokens)
        self.number_of_tokens = len(self.token_embeddings['Dev'][0][0])  # number of tokens in each sentence
        print('Number of tokens:', self.number_of_tokens)
        self.number_of_tokens = len(self.token_embeddings['Test'][0][0])  # number of tokens in each sentence
        print('Number of tokens:', self.number_of_tokens)
        print()
        self.embedding_dim = len(self.token_embeddings['Train'][0][0][0])

        if save_data is True:
            self.tokenizer = None
            self.token_transformer = None
            self._save(token_transformer_model_name)

    def __create_token_embeddings(self, sentence_pairs: list[list[str]], batch_size: int = 3) -> tuple:
        pair_of_sentences = DataManager.sentence_pairs_to_pair_of_sentences(sentence_pairs)
        all_embeddings1 = []
        all_embeddings2 = []

        tokenized_sentences1 = self.tokenizer(pair_of_sentences[0], return_tensors="pt", padding=True, truncation=True)
        tokenized_sentences2 = self.tokenizer(pair_of_sentences[1], return_tensors="pt", padding=True, truncation=True)

        for i in range(0, len(pair_of_sentences[0]), batch_size):
            # batch_sentences1 = pair_of_sentences[0][i:i + batch_size]
            # batch_sentences2 = pair_of_sentences[1][i:i + batch_size]

            # batch_sentences1 = tokenized_sentences1[i:i + batch_size]
            # batch_sentences2 = tokenized_sentences2[i:i + batch_size]

            batch_inputs1 = {
                'input_ids': tokenized_sentences1['input_ids'][i:i + batch_size],
                'attention_mask': tokenized_sentences1['attention_mask'][i:i + batch_size],
            }
            batch_inputs2 = {
                'input_ids': tokenized_sentences2['input_ids'][i:i + batch_size],
                'attention_mask': tokenized_sentences2['attention_mask'][i:i + batch_size],
            }

            # tokenized_sentences1 = self.tokenizer(batch_sentences1, return_tensors="pt", padding=True, truncation=True)
            # tokenized_sentences2 = self.tokenizer(batch_sentences2, return_tensors="pt", padding=True, truncation=True)

            # with torch.no_grad():
            #     outputs1 = self.token_transformer(**tokenized_sentences1)
            #     outputs2 = self.token_transformer(**tokenized_sentences2)

            with torch.no_grad():
                outputs1 = self.token_transformer(**batch_inputs1)
                outputs2 = self.token_transformer(**batch_inputs2)

            token_embeddings1 = outputs1.last_hidden_state
            token_embeddings2 = outputs2.last_hidden_state

            all_embeddings1.append(token_embeddings1)
            all_embeddings2.append(token_embeddings2)
            if i == 0:
                print('Number of batch sentences:', len(token_embeddings1))
                print(token_embeddings1.shape)

        max_tokens1 = max(embeddings.shape[1] for embeddings in all_embeddings1)
        max_tokens2 = max(embeddings.shape[1] for embeddings in all_embeddings2)
        print('Max tokens', max_tokens1, max_tokens2)
        print(all_embeddings1[0].shape)
        # padded_embeddings1 = [torch.nn.functional.pad(embeddings, (0, 0, max_tokens1 - embeddings.shape[1], 0), value=float('nan')) for embeddings in all_embeddings1]
        # padded_embeddings2 = [torch.nn.functional.pad(embeddings, (0, 0, max_tokens2 - embeddings.shape[1], 0), value=float('nan')) for embeddings in all_embeddings2]
        # print(padded_embeddings1[0].shape)
        # concatenated_embeddings1 = torch.cat(padded_embeddings1, dim=0)
        # print(concatenated_embeddings1.shape)
        # concatenated_embeddings2 = torch.cat(padded_embeddings2, dim=0)

        print(all_embeddings1[0].shape)
        concatenated_embeddings1 = torch.cat(all_embeddings1, dim=0)
        print(concatenated_embeddings1.shape)
        concatenated_embeddings2 = torch.cat(all_embeddings2, dim=0)
        return concatenated_embeddings1, concatenated_embeddings2

        # tokenized_sentences1 = self.tokenizer(pair_of_sentences[0][:3], return_tensors='pt', padding=True, truncation=True)
        # tokenized_sentences2 = self.tokenizer(pair_of_sentences[1][:3], return_tensors='pt', padding=True, truncation=True)
        #
        # with torch.no_grad():
        #     outputs1 = self.token_transformer(**tokenized_sentences1)
        #     outputs2 = self.token_transformer(**tokenized_sentences2)
        #
        # token_embeddings1 = outputs1.last_hidden_state
        # token_embeddings2 = outputs2.last_hidden_state
        # print(token_embeddings1)
        # return token_embeddings1, token_embeddings2

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
    def load(language: str, token_transformer_model: str = 'base uncased BERT', save_data: bool = True):
        path = 'data/token_embeddings/' + token_transformer_model + '_' + language + '.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as file:
                return pkl.load(file)
        return DataManagerWithTokenEmbeddings(language, token_transformer_model, save_data)
