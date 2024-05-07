import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.optim import Adam

from src.utilities.early_stopping import EarlyStoppingData
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
        # print('max tokens', max_len)
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
        self.name = 'Siamese LSTM (using Token Embeddings)'
        self.data = DataManagerWithTokenEmbeddings.load(language, data_split, transformer_name)

        self.model = SiameseLSTMArchitecture(self.data.embedding_dim, self.data.embedding_dim * 2)
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
            input1 = self.data.token_embeddings[dataset][0]
            input2 = self.data.token_embeddings[dataset][1]
        else:
            input1 = self.data.token_embeddings[dataset][0][batch:batch + batch_size]
            input2 = self.data.token_embeddings[dataset][1][batch:batch + batch_size]
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
        self.data.print_results(self.name, self.data.token_transformer_name, dataset)


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
