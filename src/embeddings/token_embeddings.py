from transformers import AutoTokenizer, AutoModel

from src.utilities.data_management import DataManager


class DataManagerWithTokenEmbeddings(DataManager):
    def __init__(self, language, token_transformer_model_name: str = 'bert-base-uncased'):
        super().__init__(language)
        self.tokenizer = AutoTokenizer.from_pretrained(token_transformer_model_name)
        self.token_transformer = AutoModel.from_pretrained(token_transformer_model_name)

        pass
