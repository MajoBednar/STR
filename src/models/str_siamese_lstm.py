import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.optim import Adam

from src.utilities.program_args import parse_program_args
from src.utilities.constants import Verbose, EarlyStoppingOptions as Eso
from src.embeddings.token_embeddings import DataManagerWithTokenEmbeddings
from .str_model_base import STRModelBase


class SiameseLSTM(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_layers: int = 1):
        super(SiameseLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.shared_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.common_branch = nn.Linear(hidden_dim, 1)

    def forward_lstm(self, embedding: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.shared_lstm(embedding)
        pooled_out, _ = torch.max(lstm_out, dim=1)
        return pooled_out

    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
        max_len = max(embedding1.size(1), embedding2.size(1))
        embedding1_padded = functional.pad(embedding1, (0, 0, max_len - embedding1.size(1), 0))
        embedding2_padded = functional.pad(embedding2, (0, 0, max_len - embedding2.size(1), 0))
        # forward pass of the shared branch for each input
        out1 = self.forward_lstm(embedding1_padded)
        out2 = self.forward_lstm(embedding2_padded)
        # forward pass of the common branch for the combined output
        combined = torch.abs(out1 - out2)
        return torch.sigmoid(self.common_branch(combined))


class STRSiameseLSTM(STRModelBase):
    def __init__(self, data_manager: DataManagerWithTokenEmbeddings, model: nn.Module = None,
                 learning_rate: float = 0.001, optimizer: torch.optim = None, verbose: Verbose = Verbose.DEFAULT):
        super().__init__(verbose)
        self.name: str = 'Siamese LSTM'
        self.data: DataManagerWithTokenEmbeddings = data_manager
        self.model: SiameseLSTM = SiameseLSTM(self.data.embedding_dim,
                                              self.data.embedding_dim * 2) if model is None else model
        self.model.to(self.device)
        self.optimizer: Adam = Adam(self.model.parameters(), lr=learning_rate) if optimizer is None else optimizer


def evaluate_siamese_lstm(data_manager: DataManagerWithTokenEmbeddings) -> None:
    siamese_lstm = STRSiameseLSTM(data_manager, verbose=Verbose.SILENT)
    siamese_lstm.train(epochs=10)
    siamese_lstm.evaluate()


def main() -> None:
    language, data_split = parse_program_args()
    data_manager = DataManagerWithTokenEmbeddings.load(language, data_split, 'LaBSE')

    siamese_lstm = STRSiameseLSTM(data_manager)
    siamese_lstm.train(epochs=50, early_stopping=Eso.LOSS)
    siamese_lstm.evaluate(dataset='Train')
    siamese_lstm.evaluate()


if __name__ == '__main__':
    main()
