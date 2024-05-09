import torch
import torch.nn as nn
from torch.optim import Adam

from src.utilities.program_args import parse_program_args
from src.utilities.constants import Verbose, EarlyStoppingOptions as Eso
from src.embeddings.sentence_embeddings import DataManagerWithSentenceEmbeddings
from .relatedness_model_base import RelatednessModelBase


class SiameseMLPArchitecture(nn.Module):
    def __init__(self, input_dim):
        super(SiameseMLPArchitecture, self).__init__()

        self.shared_branch = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.common_branch = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, embedding_sentence1, embedding_sentence2):
        out1 = self.shared_branch(embedding_sentence1)
        out2 = self.shared_branch(embedding_sentence2)

        combined = torch.abs(out1 - out2)
        out = torch.sigmoid(self.common_branch(combined))
        return out


class SiameseMLP(RelatednessModelBase):
    def __init__(self, language: str, data_split: str, transformer_name: str = 'all MiniLM',
                 learning_rate: float = 0.001, verbose: Verbose = Verbose.DEFAULT,
                 data_manager: DataManagerWithSentenceEmbeddings = None):
        super().__init__(verbose)
        self.name = 'Siamese MLP'
        if data_manager is None:
            self.data = DataManagerWithSentenceEmbeddings.load(language, data_split, transformer_name)
        else:
            self.data = data_manager

        self.model = SiameseMLPArchitecture(self.data.embedding_dim)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)


def evaluate_siamese_mlp(language: str, data_split: str, transformer_name: str) -> None:
    siamese_mlp = SiameseMLP(language=language, data_split=data_split, verbose=Verbose.SILENT,
                             transformer_name=transformer_name)
    siamese_mlp.train(epochs=100, early_stopping=Eso.CORR, patience=20)
    siamese_mlp.evaluate()


def main() -> None:
    language, data_split = parse_program_args()
    siamese_mlp = SiameseMLP(language, data_split, transformer_name='LaBSE')
    siamese_mlp.train(epochs=150, early_stopping=Eso.CORR, patience=20)
    siamese_mlp.evaluate(dataset='Train')
    siamese_mlp.evaluate()


if __name__ == '__main__':
    main()
