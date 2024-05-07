import torch
import torch.nn as nn

from src.utilities.constants import Verbose, EarlyStoppingOptions as Eso
from src.utilities.early_stopping import EarlyStoppingData


class RelatednessModelBase:
    def __init__(self, verbose: Verbose = Verbose.DEFAULT):
        self.verbose = verbose
        self.name = 'Relatedness Model Base'
        self.data = AbstractDataManager()

        self.model = AbstractRelatednessArchitecture()
        self.loss_function = nn.MSELoss()
        self.optimizer = AbstractOptimizer()

    def train(self, epochs: int = 1, batch_size: int = 32, early_stopping: Eso = Eso.NONE, patience: int = 20):
        early_stopping_data = EarlyStoppingData(early_stopping, patience)

        for epoch in range(epochs):
            running_loss = 0.0
            for batch in range(0, len(self.data.sentence_pairs['Train']), batch_size):
                running_loss = self.train_batch(batch, batch_size, running_loss)

            _, _, val_loss, val_correlation = self.validate('Dev')
            self._summarize_epoch(epoch, epochs, running_loss, val_loss, val_correlation)

            early_stopping_data.update(val_corr=val_correlation, val_loss=val_loss, model=self.model)
            if early_stopping_data.stop(self.verbose):
                break

        # Restore the best model state
        if early_stopping_data.best_model_state is not None:
            self.model.load_state_dict(early_stopping_data.best_model_state)

    def _summarize_epoch(self, epoch: int, epochs: int, running_loss: float, val_loss: float, val_corr: float):
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
            input1 = self.data.get_embeddings()[dataset][0]
            input2 = self.data.get_embeddings()[dataset][1]
        else:
            input1 = self.data.get_embeddings()[dataset][0][batch:batch + batch_size]
            input2 = self.data.get_embeddings()[dataset][1][batch:batch + batch_size]
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
        self.data.print_results(self.name, self.data.transformer_name, dataset)


class AbstractRelatednessArchitecture(nn.Module):
    pass


class AbstractDataManager:
    def __init__(self):
        self.transformer_name = None
        self.sentence_pairs = None
        self.scores = None

    def calculate_spearman_correlation(self, true_scores, predicted_scores) -> float:
        pass

    def set_spearman_correlation(self, true_scores, predicted_scores):
        pass

    def print_results(self, name, sentence_transformer_name, dataset):
        pass

    def get_embeddings(self) -> dict:
        pass


class AbstractOptimizer:
    def zero_grad(self):
        pass

    def step(self):
        pass
