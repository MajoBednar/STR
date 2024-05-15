import torch
import torch.nn as nn
from torch.optim import Adam

from src.utilities.program_args import parse_program_args
from src.utilities.constants import Verbose, EarlyStoppingOptions as Eso
from src.embeddings.sentence_embeddings import DataManagerWithSentenceEmbeddings
from .str_model_base import STRModelBase


class SiameseMLP(nn.Module):
    def __init__(self, input_dim):
        super(SiameseMLP, self).__init__()

        self.shared_branch = nn.Sequential(nn.Linear(input_dim, 1024), nn.ReLU(),
                                           nn.Linear(1024, 512), nn.ReLU(),
                                           nn.Linear(512, 256), nn.ReLU(),
                                           nn.Linear(256, 128), nn.ReLU())

        self.common_branch = nn.Sequential(nn.Linear(128, 32), nn.ReLU(),
                                           nn.Linear(32, 1))

    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
        out1 = self.shared_branch(embedding1)
        out2 = self.shared_branch(embedding2)

        combined = torch.abs(out1 - out2)
        return torch.sigmoid(self.common_branch(combined))


class STRSiameseMLP(STRModelBase):
    def __init__(self, data_manager: DataManagerWithSentenceEmbeddings, learning_rate: float = 0.001,
                 verbose: Verbose = Verbose.DEFAULT):
        super().__init__(verbose)
        self.name: str = 'Siamese MLP'
        self.data: DataManagerWithSentenceEmbeddings = data_manager
        self.model: SiameseMLP = SiameseMLP(self.data.embedding_dim)
        self.optimizer: Adam = Adam(self.model.parameters(), lr=learning_rate)


def evaluate_siamese_mlp(data_manager: DataManagerWithSentenceEmbeddings) -> None:
    siamese_mlp = STRSiameseMLP(data_manager, verbose=Verbose.SILENT)
    siamese_mlp.train(epochs=100, early_stopping=Eso.CORR, patience=20)
    siamese_mlp.evaluate()


def main() -> None:
    language, data_split = parse_program_args()
    data_manager = DataManagerWithSentenceEmbeddings.load(language, data_split, 'LaBSE')

    siamese_mlp = STRSiameseMLP(data_manager)
    siamese_mlp.train(epochs=150, early_stopping=Eso.CORR, patience=20)
    siamese_mlp.evaluate(dataset='Train')
    siamese_mlp.evaluate()


if __name__ == '__main__':
    main()
