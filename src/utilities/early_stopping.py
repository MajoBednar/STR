from src.utilities.constants import Verbose, EarlyStoppingOptions as Eso
import torch.nn as nn


class EarlyStoppingData:
    def __init__(self, early_stopping: Eso, patience: int):
        self.early_stopping: Eso = early_stopping
        self.patience = patience
        self.best_val_correlation = -1
        self.best_val_loss = float('inf')
        self.no_improvement_count = 0
        self.best_model_state = None

    def update(self, val_corr: float, val_loss: float, model: nn.Module) -> None:
        if self.early_stopping == Eso.NONE:
            pass
        elif self.early_stopping == Eso.LOSS:
            self.__update_loss(val_loss, model)
        elif self.early_stopping == Eso.CORR:
            self.__update_corr(val_corr, model)

    def stop(self, verbose: Verbose = Verbose.DEFAULT) -> bool:
        if self.no_improvement_count >= self.patience:
            if verbose == Verbose.DEFAULT or verbose == Verbose.EXPRESSIVE:
                print(f'No improvement for {self.patience} epochs. Early stopping.')
            return True
        else:
            return False

    def __update_loss(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.no_improvement_count = 0
            self.best_model_state = model.state_dict()
        else:
            self.no_improvement_count += 1

    def __update_corr(self, val_correlation: float, model: nn.Module) -> None:
        if val_correlation > self.best_val_correlation:
            self.best_val_correlation = val_correlation
            self.no_improvement_count = 0
            self.best_model_state = model.state_dict()
        else:
            self.no_improvement_count += 1
