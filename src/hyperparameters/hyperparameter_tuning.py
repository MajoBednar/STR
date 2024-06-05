import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop, SGD
import optuna


def find_best_transformers(model: any, transformer: str, best_transformers: dict[str, str],
                           best_correlations: dict[str, float]) -> tuple[dict[str, str], dict[str, float]]:
    corr_train = model.evaluate('Train')
    corr_dev = model.evaluate('Dev')
    corr_test = model.evaluate('Test')

    if corr_train > best_correlations['Train']:
        best_correlations['Train'] = corr_train
        best_transformers['Train'] = transformer
    if corr_dev > best_correlations['Dev']:
        best_correlations['Dev'] = corr_dev
        best_transformers['Dev'] = transformer
    if corr_test > best_correlations['Test']:
        best_correlations['Test'] = corr_test
        best_transformers['Test'] = transformer

    return best_transformers, best_correlations


def hyperparams_for_optimizer(trial: optuna.trial) -> tuple[any, any, any]:
    optimizer_name = trial.suggest_categorical('optimizer', ('Adam', 'SGD', 'RMSprop'))
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    return optimizer_name, learning_rate, weight_decay


def hyperparams_for_early_stopping(trial: optuna.trial) -> tuple[any, any]:
    early_stopping_option = trial.suggest_categorical('early_stopping_option', (0, 1, 2))
    patience = trial.suggest_categorical('patience', (20, 30, 100))
    return early_stopping_option, patience


def print_study_results(study: optuna.Study) -> None:
    print('\nBest hyperparameters found:')
    print(study.best_params)
    print('With validation correlation:', study.best_value)


def get_activation(activation_name: str) -> ():
    if activation_name == 'ReLU':
        return nn.ReLU
    elif activation_name == 'LeakyReLU':
        return nn.LeakyReLU
    elif activation_name == 'Tanh':
        return nn.Tanh
    elif activation_name == 'Sigmoid':
        return nn.Sigmoid
    else:
        raise ValueError(f'Unknown activation function: {activation_name}')


def get_optimizer(optimizer_name: str, model_architecture: nn.Module, learning_rate: float,
                  weight_decay: float) -> torch.optim:
    if optimizer_name == 'Adam':
        return Adam(model_architecture.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        return SGD(model_architecture.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        return RMSprop(model_architecture.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')


def get_shared_layer_sizes_mlp(architecture_size: str) -> tuple:
    if architecture_size == 'Small':
        return 512, 256, 128
    elif architecture_size == 'Medium':
        return 1024, 512, 256
    elif architecture_size == 'Big':
        return 1024, 512, 256, 128
    else:
        raise ValueError(f'Unknown architecture size: {architecture_size}')


def get_common_layer_sizes_mlp(architecture_size: str) -> tuple:
    if architecture_size == 'Small':
        return 32, 1
    elif architecture_size == 'Medium':
        return 64, 32, 1
    elif architecture_size == 'Big':
        return 128, 64, 32, 1
    else:
        raise ValueError(f'Unknown architecture size: {architecture_size}')
