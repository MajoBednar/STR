from transformers import AutoTokenizer, AutoModel
import torch
import os
import pickle as pkl

from src.utilities.data_management import DataManager
from src.utilities.constants import TOKEN_TRANSFORMERS as TT


class DataManagerWithTokenEmbeddings(DataManager):
    def __init__(self, language: str, data_split: str, token_transformer_model_name: str, save_data: bool):
        super().__init__(language, data_split)
        self.transformer_name: str = token_transformer_model_name + ' Token Transformer'

        tokenizer = AutoTokenizer.from_pretrained(TT[token_transformer_model_name])
        token_transformer = AutoModel.from_pretrained(TT[token_transformer_model_name])
        self.token_embeddings: dict[str, tuple[iter, iter]] = {
            'Train': self.__create_token_embeddings(self.sentence_pairs['Train'], tokenizer, token_transformer),
            'Dev': self.__create_token_embeddings(self.sentence_pairs['Dev'], tokenizer, token_transformer),
            'Test': self.__create_token_embeddings(self.sentence_pairs['Test'], tokenizer, token_transformer)
        }
        self.embedding_dim: int = len(self.token_embeddings['Train'][0][0][0])

        if save_data is True:
            self._save(token_transformer_model_name)

    def get_embeddings(self) -> dict[str, tuple[iter, iter]]:
        return self.token_embeddings

    @staticmethod
    def __create_token_embeddings(sentence_pairs: list[list[str]], tokenizer: any, token_transformer: any,
                                  batch_size: int = 32) -> tuple[iter, iter]:
        pair_of_sentences = DataManager.sentence_pairs_to_pair_of_sentences(sentence_pairs)
        all_embeddings1, all_embeddings2 = [], []
        # tokenize sentences
        tokenized_sentences1 = tokenizer(pair_of_sentences[0], return_tensors='pt', padding=True, truncation=True)
        tokenized_sentences2 = tokenizer(pair_of_sentences[1], return_tensors='pt', padding=True, truncation=True)
        # process in batches
        for i in range(0, len(pair_of_sentences[0]), batch_size):
            batch_inputs1 = {
                'input_ids': tokenized_sentences1['input_ids'][i:i + batch_size],
                'attention_mask': tokenized_sentences1['attention_mask'][i:i + batch_size],
            }
            batch_inputs2 = {
                'input_ids': tokenized_sentences2['input_ids'][i:i + batch_size],
                'attention_mask': tokenized_sentences2['attention_mask'][i:i + batch_size],
            }
            # forward pass
            with torch.no_grad():
                outputs1 = token_transformer(**batch_inputs1)
                outputs2 = token_transformer(**batch_inputs2)
            # get token embeddings
            token_embeddings1 = outputs1.last_hidden_state
            token_embeddings2 = outputs2.last_hidden_state
            # add embeddings to list
            all_embeddings1.append(token_embeddings1)
            all_embeddings2.append(token_embeddings2)
            print(i)
        # concatenate embeddings into one tensor
        concatenated_embeddings1 = torch.cat(all_embeddings1, dim=0)
        concatenated_embeddings2 = torch.cat(all_embeddings2, dim=0)
        return concatenated_embeddings1, concatenated_embeddings2

    def _save(self, transformer_model: str, directory: str = 'data/token_embeddings/') -> None:
        super()._save(transformer_model, directory)

    @staticmethod
    def load(language: str, data_split: str, token_transformer_model: str, save_data: bool = True):
        # load existing data manager
        path = 'data/token_embeddings/' + token_transformer_model + '_' + language + '_' + data_split + '.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as file:
                return pkl.load(file)
        # create data manager if not found
        return DataManagerWithTokenEmbeddings(language, data_split, token_transformer_model, save_data)
