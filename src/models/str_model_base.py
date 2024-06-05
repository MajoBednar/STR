import torch
import torch.nn as nn

from src.utilities.constants import Verbose, EarlyStoppingOptions as Eso
from src.utilities.early_stopping import EarlyStoppingData


class STRModelBase:
    def __init__(self, verbose: Verbose = Verbose.DEFAULT):
        self.verbose: Verbose = verbose
        self.name: str = 'Semantic Text Relatedness Model Base'
        self.data: AbstractDataManager = AbstractDataManager()

        self.model: AbstractModelArchitecture = AbstractModelArchitecture()
        self.loss_function: nn.MSELoss = nn.MSELoss()
        self.optimizer: AbstractOptimizer = AbstractOptimizer()

        # Move the model to GPU if available
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('The model will be using device:', self.device)
        self.model.to(self.device)

    def train(self, epochs: int = 1, batch_size: int = 32, early_stopping: Eso = Eso.NONE, patience: int = 20) -> None:
        early_stopping_data = EarlyStoppingData(early_stopping, patience)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for batch in range(0, len(self.data.sentence_pairs['Train']), batch_size):
                running_loss = self.train_batch(batch, batch_size, running_loss)

            _, _, val_loss, val_correlation = self.validate('Dev')
            self._summarize_epoch(epoch, epochs, running_loss, val_loss, val_correlation)
            # update early stopping conditions
            early_stopping_data.update(val_corr=val_correlation, val_loss=val_loss, model=self.model)
            if early_stopping_data.stop(self.verbose):
                break

        if early_stopping_data.best_model_state is not None:  # restore the best model state
            self.model.load_state_dict(early_stopping_data.best_model_state)

    def _summarize_epoch(self, epoch: int, epochs: int, running_loss: float, val_loss: float, val_corr: float) -> None:
        epoch_loss = running_loss / len(self.data.sentence_pairs['Train'])
        if self.verbose == Verbose.DEFAULT or self.verbose == Verbose.EXPRESSIVE:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}', end='')
            print(f', Val Correlation: {val_corr:.4f}')
        if self.verbose == Verbose.EXPRESSIVE:
            self.evaluate()

    def train_batch(self, batch: int, batch_size: int, running_loss: float) -> float:
        self.optimizer.zero_grad()

        predicted_scores = self.predict('Train', batch, batch_size)
        true_scores = torch.tensor(self.data.scores['Train'][batch:batch + batch_size]).unsqueeze(1).to(self.device)

        loss = self.loss_function(predicted_scores, true_scores)
        loss.backward()
        self.optimizer.step()

        return running_loss + loss.item() * predicted_scores.size(0)  # increase and return running loss

    def predict(self, dataset: str, batch: int = 0, batch_size: int = 0) -> torch.Tensor:
        if batch_size == 0:  # predict the score for each data point
            input1 = self.data.get_embeddings()[dataset][0].to(self.device)
            input2 = self.data.get_embeddings()[dataset][1].to(self.device)
        else:  # predict the score for data points in a batch
            input1 = self.data.get_embeddings()[dataset][0][batch:batch + batch_size].to(self.device)
            input2 = self.data.get_embeddings()[dataset][1][batch:batch + batch_size].to(self.device)
        return self.model(input1, input2)

    def validate(self, dataset: str) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        self.model.eval()
        with torch.no_grad():
            predicted_scores = self.predict(dataset)
            true_scores = torch.tensor(self.data.scores[dataset]).unsqueeze(1).to(self.device)
            loss = self.loss_function(predicted_scores, true_scores)
            correlation = self.data.calculate_spearman_correlation(true_scores, predicted_scores)
        return predicted_scores, true_scores, loss, correlation

    def evaluate(self, dataset: str = 'Test') -> float:
        predicted_scores, true_scores, _, correlation = self.validate(dataset)
        if self.verbose == Verbose.EXPRESSIVE:
            print(predicted_scores)

        self.data.print_results(correlation, self.name, self.data.transformer_name, dataset)
        return correlation


class AbstractModelArchitecture(nn.Module):
    pass


class AbstractDataManager:
    def __init__(self):
        self.transformer_name = None
        self.sentence_pairs = None
        self.scores = None

    def calculate_spearman_correlation(self, true_scores, predicted_scores) -> float:
        pass

    def print_results(self, correlation, name, sentence_transformer_name, dataset) -> None:
        pass

    def get_embeddings(self) -> dict[str, tuple[iter, iter]]:
        pass


class AbstractOptimizer:
    def zero_grad(self) -> None:
        pass

    def step(self) -> None:
        pass
