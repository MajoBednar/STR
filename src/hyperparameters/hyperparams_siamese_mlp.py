import optuna

from src.utilities.program_args import parse_program_args
from src.utilities.constants import SENTENCE_TRANSFORMERS
from src.embeddings.sentence_embeddings import DataManagerWithSentenceEmbeddings
from src.models.str_siamese_mlp import STRSiameseMLP

"""Hyperparameters for Siamese MLP: 
Architecture: shared layer, common layer, activation function, dropout
Transformer;
Optimizer: type, learning rate;
Early stopping: type, patience;
Training: epochs, batch size; """


def objective(trial):
    # Hyperparameters to tune
    input_dim = 300
    shared_layer_sizes = trial.suggest_categorical('shared_layer_sizes',
                                                   [[1024, 512, 256, 128], [512, 256, 128], [1024, 512, 256]])
    common_layer_sizes = trial.suggest_categorical('common_layer_sizes', [[32, 1], [64, 32, 1], [128, 64, 32, 1]])
    activation = trial.suggest_categorical('activation', [nn.ReLU, nn.LeakyReLU])
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    num_epochs = trial.suggest_int('num_epochs', 10, 50)
    gradient_clip = trial.suggest_float('gradient_clip', 0.0, 1.0)

    model = SiameseMLP(input_dim=input_dim,
                       shared_layer_sizes=shared_layer_sizes,
                       common_layer_sizes=common_layer_sizes,
                       activation=activation,
                       dropout=dropout)

    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    criterion = nn.BCELoss()

    train_loader = DataLoader(MyDataset(train_data, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MyDataset(val_data, val_labels), batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data[0], data[1])
            loss = criterion(outputs, labels)
            loss.backward()
            if gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, labels in val_loader:
            outputs = model(data[0], data[1])
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    return val_loss


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print(study.best_params)
