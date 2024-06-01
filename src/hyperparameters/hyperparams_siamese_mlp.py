import optuna
import torch

from src.utilities.program_args import parse_program_args
from src.utilities.constants import SENTENCE_TRANSFORMERS, Verbose
from src.embeddings.sentence_embeddings import DataManagerWithSentenceEmbeddings
from src.models.str_siamese_mlp import SiameseMLP, STRSiameseMLP
from .hyperparameter_tuning import get_activation, get_optimizer, get_shared_layer_sizes_mlp, get_common_layer_sizes_mlp

""" Hyperparameters for Siamese MLP: 
Transformer;
Architecture: shared layers, common layers, activation function, dropout
Optimizer: type, learning rate, weight decay;
Early stopping: type, patience;
Training: epochs, batch size; 
Optional: gradient clipping; """


def objective(trial: optuna.trial, language: str, data_split: str):
    # Hyperparameters to tune
    transformer = trial.suggest_categorical('transformer', [name for name in SENTENCE_TRANSFORMERS])
    shared_layers_size = trial.suggest_categorical('shared_layers_size', ('Small', 'Medium', 'Big'))
    common_layers_size = trial.suggest_categorical('common_layers_size', ('Small', 'Medium', 'Big'))
    activation_name = trial.suggest_categorical('activation', ('ReLU', 'LeakyReLU'))
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    optimizer_name = trial.suggest_categorical('optimizer', ('Adam', 'SGD', 'RMSprop'))
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    early_stopping_option = trial.suggest_categorical('early_stopping_option', (0, 1, 2))
    patience = trial.suggest_categorical('patience', (20, 30, 100))
    batch_size = trial.suggest_categorical('batch_size', (16, 32, 64))
    num_epochs = trial.suggest_int('num_epochs', 1, 200)
    # gradient_clip = trial.suggest_float('gradient_clip', 0.0, 1.0)

    # Setup parameters for the Siamese MLP model
    data_manager = DataManagerWithSentenceEmbeddings.load(language, data_split, transformer)
    shared_layer_sizes = get_shared_layer_sizes_mlp(shared_layers_size)
    common_layer_sizes = get_common_layer_sizes_mlp(common_layers_size)
    activation = get_activation(activation_name)
    model_architecture = SiameseMLP(input_dim=data_manager.embedding_dim,
                                    shared_layer_sizes=shared_layer_sizes,
                                    common_layer_sizes=common_layer_sizes,
                                    activation=activation,
                                    dropout=dropout)
    model_architecture = torch.jit.script(model_architecture)
    optimizer = get_optimizer(optimizer_name, model_architecture, learning_rate, weight_decay)

    # Train and evaluate model with chosen hyperparameters
    model = STRSiameseMLP(data_manager, model_architecture, learning_rate, optimizer, Verbose.SILENT)
    model.train(num_epochs, batch_size, early_stopping_option, patience)

    # after loss.backward() and before optimizer.step()
    # if gradient_clip > 0:
    #     nn.utils.clip_grad_norm_(model_architecture.parameters(), gradient_clip)

    _, _, val_loss, val_corr = model.validate('Dev')
    return val_corr


def main():
    language, data_split = parse_program_args()
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, language, data_split), n_trials=3)
    print('\nBest hyperparameters found:')
    best_params = study.best_params
    print(best_params)
    print('With validation correlation:', study.best_value)
    # Evaluate the best hyperparameters on the test set
    data_manager = DataManagerWithSentenceEmbeddings.load(language, data_split, best_params['transformer'])
    shared_layer_sizes = get_shared_layer_sizes_mlp(best_params['shared_layers_size'])
    common_layer_sizes = get_common_layer_sizes_mlp(best_params['common_layers_size'])
    activation = get_activation(best_params['activation'])
    model_architecture = SiameseMLP(input_dim=data_manager.embedding_dim,
                                    shared_layer_sizes=shared_layer_sizes,
                                    common_layer_sizes=common_layer_sizes,
                                    activation=activation,
                                    dropout=best_params['dropout'])
    model_architecture = torch.jit.script(model_architecture)
    optimizer = get_optimizer(best_params['optimizer'], model_architecture, best_params['learning_rate'],
                              best_params['weight_decay'])

    # Train and evaluate model with chosen hyperparameters
    model = STRSiameseMLP(data_manager, model_architecture, best_params['learning_rate'], optimizer, Verbose.SILENT)
    model.train(best_params['num_epochs'], best_params['batch_size'], best_params['early_stopping_option'],
                best_params['patience'])
    model.evaluate()


if __name__ == '__main__':
    main()
