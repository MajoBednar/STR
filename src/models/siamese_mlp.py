import torch
import torch.nn as nn
from torch.optim import Adam

from src.utilities.program_args import parse_program_args
from src.utilities.constants import Verbose, EarlyStoppingOptions as Eso
from src.utilities.early_stopping import EarlyStoppingData
from src.embeddings.sentence_embeddings import DataManagerWithSentenceEmbeddings
from .relatedness_model_base import RelatednessModelBase


class SiameseMLPArchitecture(nn.Module):
    def __init__(self, input_dim):
        super(SiameseMLPArchitecture, self).__init__()

        self.shared_branch = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
            # nn.Linear(512, 128),
            # nn.ReLU()
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
                 learning_rate: float = 0.001, verbose: Verbose = Verbose.DEFAULT):
        super().__init__(verbose)
        self.name = 'Siamese MLP'
        self.data = DataManagerWithSentenceEmbeddings.load(language, data_split, transformer_name)

        self.model = SiameseMLPArchitecture(self.data.embedding_dim)
        self.loss_function = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

    def train(self, epochs: int = 1, batch_size: int = 32, early_stopping: Eso = Eso.NONE, patience: int = 20):
        early_stopping_data = EarlyStoppingData(early_stopping, patience)

        for epoch in range(epochs):
            running_loss = 0.0
            for batch in range(0, len(self.data.sentence_pairs['Train']), batch_size):
                running_loss = self.train_batch(batch, batch_size, running_loss)

            _, _, val_loss, val_correlation = self.validate('Dev')
            self.__summarize_epoch(epoch, epochs, running_loss, val_loss, val_correlation)

            early_stopping_data.update(val_corr=val_correlation, val_loss=val_loss, model=self.model)
            if early_stopping_data.stop(self.verbose):
                break

        # Restore the best model state
        if early_stopping_data.best_model_state is not None:
            self.model.load_state_dict(early_stopping_data.best_model_state)

    def __summarize_epoch(self, epoch: int, epochs: int, running_loss: float, val_loss: float, val_corr: float):
        epoch_loss = running_loss / len(self.data.sentence_pairs['Train'])
        if self.verbose == Verbose.DEFAULT or self.verbose == Verbose.EXPRESSIVE:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}', end='')
            print(f', Val Correlation: {val_corr:.4f}')
        if self.verbose == Verbose.EXPRESSIVE:
            self.evaluate()

    def train_batch(self, batch: int, batch_size: int, running_loss: float) -> float:
        self.optimizer.zero_grad()

        predicted_scores = self.predict('Train', batch, batch_size)
        true_scores = torch.tensor(self.data.scores['Train'][batch:batch + batch_size]).unsqueeze(1)

        loss = self.loss_function(predicted_scores, true_scores)
        loss.backward()
        self.optimizer.step()

        running_loss += loss.item() * predicted_scores.size(0)
        return running_loss

    def predict(self, dataset: str, batch: int = 0, batch_size: int = 0):
        if batch_size == 0:
            input1 = self.data.sentence_embeddings[dataset][0]
            input2 = self.data.sentence_embeddings[dataset][1]
        else:
            input1 = self.data.sentence_embeddings[dataset][0][batch:batch + batch_size]
            input2 = self.data.sentence_embeddings[dataset][1][batch:batch + batch_size]
        return self.model(input1, input2)

    def validate(self, dataset: str) -> tuple:
        with torch.no_grad():
            predicted_scores = self.predict(dataset)
            true_scores = torch.tensor(self.data.scores[dataset]).unsqueeze(1)
            loss = self.loss_function(predicted_scores, true_scores)
            correlation = self.data.calculate_spearman_correlation(true_scores, predicted_scores)
        return predicted_scores, true_scores, loss, correlation

    def evaluate(self, dataset: str = 'Test'):
        predicted_scores, true_scores, _, _ = self.validate(dataset)

        if self.verbose == Verbose.EXPRESSIVE:
            print(predicted_scores)

        self.data.set_spearman_correlation(true_scores, predicted_scores)
        self.data.print_results(self.name, self.data.sentence_transformer_name, dataset)


def evaluate_siamese_mlp(language: str, data_split: str, transformer_name: str) -> None:
    siamese_mlp = SiameseMLP(language=language, data_split=data_split, verbose=Verbose.SILENT,
                             transformer_name=transformer_name)
    siamese_mlp.train(epochs=100, early_stopping=Eso.CORR, patience=20)
    siamese_mlp.evaluate()


def main() -> None:
    language, data_split = parse_program_args()
    siamese_mlp = SiameseMLP(language, data_split, transformer_name='LaBSE')
    siamese_mlp.train(epochs=50, early_stopping=Eso.CORR, patience=20)
    siamese_mlp.evaluate(dataset='Train')
    siamese_mlp.evaluate()


if __name__ == '__main__':
    main()
