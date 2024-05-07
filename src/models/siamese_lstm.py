import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.optim import Adam

from src.utilities.program_args import parse_program_args
from src.utilities.constants import Verbose, EarlyStoppingOptions as Eso
from src.embeddings.token_embeddings import DataManagerWithTokenEmbeddings
from .relatedness_model_base import RelatednessModelBase


class SiameseLSTMArchitecture(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=1):
        super(SiameseLSTMArchitecture, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, 1)

    def forward_one(self, x):
        lstm_out, _ = self.lstm(x)
        pooled_out, _ = torch.max(lstm_out, dim=1)
        return pooled_out

    def forward(self, x1, x2):
        max_len = max(x1.size(1), x2.size(1))
        x1_padded = functional.pad(x1, (0, 0, max_len - x1.size(1), 0))
        x2_padded = functional.pad(x2, (0, 0, max_len - x2.size(1), 0))

        out1 = self.forward_one(x1_padded)
        out2 = self.forward_one(x2_padded)

        combined = torch.abs(out1 - out2)
        relatedness_score = torch.sigmoid(self.fc(combined))
        return relatedness_score


class SiameseLSTM(RelatednessModelBase):
    def __init__(self, language: str, data_split: str, transformer_name: str = 'base uncased BERT',
                 learning_rate: float = 0.001, verbose: Verbose = Verbose.DEFAULT):
        super().__init__(verbose)
        self.name = 'Siamese LSTM'
        self.data = DataManagerWithTokenEmbeddings.load(language, data_split, transformer_name)

        self.model = SiameseLSTMArchitecture(self.data.embedding_dim, self.data.embedding_dim * 2)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)


def evaluate_siamese_lstm(language: str, data_split: str, transformer_name: str) -> None:
    siamese_lstm = SiameseLSTM(language=language, data_split=data_split, verbose=Verbose.SILENT,
                               transformer_name=transformer_name)
    siamese_lstm.train(epochs=10)
    siamese_lstm.evaluate()


def main() -> None:
    language, data_split = parse_program_args()
    siamese_lstm = SiameseLSTM(language, data_split, 'LaBSE')
    # print('Embedding dim:', siamese_lstm.data.embedding_dim)
    # print('Number of tokens in Train set 1st sentence from each pair:',
    #       len(siamese_lstm.data.token_embeddings['Train'][0][0]))
    # print('Number of tokens in Train set 2nd sentence from each pair:',
    #       len(siamese_lstm.data.token_embeddings['Train'][1][0]))
    # print('Number of tokens in Dev set 1st sentence from each pair:',
    #       len(siamese_lstm.data.token_embeddings['Dev'][0][0]))
    # print('Number of tokens in Dev set 2nd sentence from each pair:',
    #       len(siamese_lstm.data.token_embeddings['Dev'][1][0]))
    # print('Number of tokens in Test set 1st sentence from each pair:',
    #       len(siamese_lstm.data.token_embeddings['Test'][0][0]))
    # print('Number of tokens in Test set 2nd sentence from each pair:',
    #       len(siamese_lstm.data.token_embeddings['Test'][1][0]))
    siamese_lstm.train(epochs=2, early_stopping=Eso.CORR)
    siamese_lstm.evaluate(dataset='Train')
    siamese_lstm.evaluate()


if __name__ == '__main__':
    main()
