import optuna
import torch

from src.utilities.program_args import parse_program_args
from src.utilities.constants import TOKEN_TRANSFORMERS, Verbose
from src.embeddings.token_embeddings import DataManagerWithTokenEmbeddings
from src.models.str_siamese_lstm import SiameseLSTM, STRSiameseLSTM
from .hyperparameter_tuning import hyperparams_for_optimizer, print_study_results, get_optimizer

""" Hyperparameters for Siamese LSTM: 
Transformer;
Architecture: hidden dimension, number of layers, (optional: dropout);
Optimizer: type, learning rate, weight decay;
Early stopping: type, patience;
Training: epochs, batch size; 
"""


def objective(trial: optuna.trial, language: str, data_split: str):
    # Hyperparameters to tune
    transformer = trial.suggest_categorical('transformer', [name for name in TOKEN_TRANSFORMERS])
    hidden_dim_factor = trial.suggest_categorical('hidden_dim_factor', (1/2, 1, 2, 3))
    num_layers = trial.suggest_categorical('num_layers', (1, 2, 3))
    optimizer_name, learning_rate, weight_decay = hyperparams_for_optimizer(trial)
    early_stopping_option = trial.suggest_categorical('early_stopping_option', (0, 1, 2))
    patience = trial.suggest_categorical('patience', (20, 30, 100))
    batch_size = trial.suggest_categorical('batch_size', (16, 32, 64))
    num_epochs = trial.suggest_int('num_epochs', 1, 100)

    # Setup parameters for the Siamese LSTM model
    data_manager = DataManagerWithTokenEmbeddings.load(language, data_split, transformer)
    model_architecture = SiameseLSTM(data_manager.embedding_dim, int(data_manager.embedding_dim * hidden_dim_factor),
                                     num_layers)
    model_architecture = torch.jit.script(model_architecture)
    optimizer = get_optimizer(optimizer_name, model_architecture, learning_rate, weight_decay)

    # Train and evaluate model with chosen hyperparameters
    model = STRSiameseLSTM(data_manager, model_architecture, learning_rate, optimizer, Verbose.SILENT)
    model.train(num_epochs, batch_size, early_stopping_option, patience)

    _, _, val_loss, val_corr = model.validate('Dev')
    return val_corr


def main():
    language, data_split = parse_program_args()
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, language, data_split), n_trials=100)
    best_params = study.best_params
    print_study_results(study)
    # Evaluate the best hyperparameters on the test set
    data_manager = DataManagerWithTokenEmbeddings.load(language, data_split, best_params['transformer'])
    hidden_dim = int(data_manager.embedding_dim * best_params['hidden_dim_factor'])
    model_architecture = SiameseLSTM(data_manager.embedding_dim, hidden_dim, best_params['num_layers'])
    model_architecture = torch.jit.script(model_architecture)
    optimizer = get_optimizer(best_params['optimizer'], model_architecture, best_params['learning_rate'],
                              best_params['weight_decay'])

    # Train and evaluate model with chosen hyperparameters
    model = STRSiameseLSTM(data_manager, model_architecture, best_params['learning_rate'], optimizer, Verbose.SILENT)
    model.train(best_params['num_epochs'], best_params['batch_size'], best_params['early_stopping_option'],
                best_params['patience'])
    model.evaluate('Train')
    model.evaluate('Dev')
    model.evaluate()


if __name__ == '__main__':
    main()
