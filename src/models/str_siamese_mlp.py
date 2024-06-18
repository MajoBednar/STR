import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop

from src.utilities.program_args import parse_program_args
from src.utilities.constants import Verbose, EarlyStoppingOptions as Eso
from src.embeddings.sentence_embeddings import DataManagerWithSentenceEmbeddings
from .str_model_base import STRModelBase


class SiameseMLP(nn.Module):
    def __init__(self, input_dim: int, shared_layer_sizes: tuple = (1024, 512, 256, 128),
                 common_layer_sizes: tuple = (32, 1), activation: () = nn.ReLU, dropout: float = 0.0):
        super(SiameseMLP, self).__init__()
        # Create shared branch
        shared_layers = []
        prev_size = input_dim
        for size in shared_layer_sizes:
            shared_layers.append(nn.Linear(prev_size, size))
            shared_layers.append(activation())
            if dropout > 0:
                shared_layers.append(nn.Dropout(dropout))
            prev_size = size
        self.shared_branch = nn.Sequential(*shared_layers)
        # Create common branch
        common_layers = []
        prev_size = shared_layer_sizes[-1]
        for size in common_layer_sizes:
            common_layers.append(nn.Linear(prev_size, size))
            if size != 1:  # Apply activation and dropout to all but the last layer
                common_layers.append(activation())
                if dropout > 0:
                    common_layers.append(nn.Dropout(dropout))
            prev_size = size
        self.common_branch = nn.Sequential(*common_layers)

    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
        out1 = self.shared_branch(embedding1)
        out2 = self.shared_branch(embedding2)

        combined = torch.abs(out1 - out2)
        return torch.sigmoid(self.common_branch(combined))


class STRSiameseMLP(STRModelBase):
    def __init__(self, data_manager: DataManagerWithSentenceEmbeddings, model: nn.Module = None,
                 learning_rate: float = 0.001, optimizer: torch.optim = None, verbose: Verbose = Verbose.DEFAULT):
        super().__init__(verbose)
        self.name: str = 'Siamese MLP'
        self.data: DataManagerWithSentenceEmbeddings = data_manager
        self.model: SiameseMLP = SiameseMLP(self.data.embedding_dim) if model is None else model
        self.model.to(self.device)
        self.optimizer: Adam = Adam(self.model.parameters(), lr=learning_rate) if optimizer is None else optimizer


def evaluate_siamese_mlp(data_manager: DataManagerWithSentenceEmbeddings) -> None:
    siamese_mlp = STRSiameseMLP(data_manager, verbose=Verbose.SILENT)
    siamese_mlp.train(epochs=100, early_stopping=Eso.CORR, patience=20)
    siamese_mlp.evaluate()


def main() -> None:
    language, data_split = parse_program_args()
    data_manager = DataManagerWithSentenceEmbeddings.load(language, data_split, 'mBERT')

    architecture = SiameseMLP(data_manager.embedding_dim, (512, 256, 128), (128, 64, 32, 1), nn.LeakyReLU, 0.287)
    optimizer = Adam(architecture.parameters(), lr=0.000674, weight_decay=1.3e-5)

    siamese_mlp = STRSiameseMLP(data_manager, architecture, 0.000674, optimizer)
    siamese_mlp.train(epochs=154, batch_size=32, early_stopping=Eso.CORR, patience=20)
    siamese_mlp.evaluate(dataset='Train')
    siamese_mlp.evaluate(dataset='Dev')
    siamese_mlp.evaluate()


if __name__ == '__main__':
    main()
