import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.optim import Adam

from src.utilities.program_args import parse_program_args
from src.utilities.constants import Verbose
from src.embeddings.token_embeddings import DataManagerWithTokenEmbeddings


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
        x1_padded = functional.pad(x1, (0, 0, max_len - x1.size(1), 0), value=float('nan'))
        x2_padded = functional.pad(x2, (0, 0, max_len - x2.size(1), 0), value=float('nan'))

        out1 = self.forward_one(x1_padded)
        out2 = self.forward_one(x2_padded)

        combined = torch.abs(out1 - out2)
        relatedness_score = torch.sigmoid(self.fc(combined))
        return relatedness_score


class SiameseLSTM:
    def __init__(self, language: str, learning_rate: float = 0.001, verbose: Verbose = Verbose.DEFAULT):
        self.name = 'Siamese LSTM (using Token Embeddings)'
        self.data = DataManagerWithTokenEmbeddings.load(language)
        self.verbose: Verbose = verbose

        self.model = SiameseLSTMArchitecture(self.data.embedding_dim, self.data.embedding_dim * 2)
        self.loss_function = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

    def train(self, epochs: int = 1, batch_size: int = 32):
        for epoch in range(epochs):
            running_loss = 0.0
            for batch in range(0, len(self.data.sentence_pairs['Train']), batch_size):
                self.optimizer.zero_grad()
                inputs1 = self.data.token_embeddings['Train'][0][batch:batch + batch_size]
                inputs2 = self.data.token_embeddings['Train'][1][batch:batch + batch_size]
                true_scores = torch.tensor(self.data.scores['Train'][batch:batch + batch_size])

                outputs = self.model(inputs1, inputs2)
                print(outputs)
                true_scores = true_scores.unsqueeze(1)
                loss = self.loss_function(outputs, true_scores)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs1.size(0)

            epoch_loss = running_loss / len(self.data.sentence_pairs['Train'])

            input1 = torch.tensor(self.data.token_embeddings['Test'][0])
            input2 = torch.tensor(self.data.token_embeddings['Test'][1])
            true_scores_test = torch.tensor(self.data.scores['Test'])
            true_scores_test = true_scores_test.unsqueeze(1)
            with torch.no_grad():
                predicted_scores_test = self.model(input1, input2)
                test_loss = self.loss_function(predicted_scores_test, true_scores_test)
            if self.verbose == Verbose.DEFAULT or self.verbose == Verbose.EXPRESSIVE:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")
            if self.verbose == Verbose.EXPRESSIVE:
                self.evaluate()

    def evaluate(self, dataset: str = 'Test'):
        input1 = torch.tensor(self.data.token_embeddings[dataset][0])
        input2 = torch.tensor(self.data.token_embeddings[dataset][1])
        with torch.no_grad():
            predicted_scores = self.model(input1, input2)

        if self.verbose == Verbose.EXPRESSIVE:
            print(predicted_scores)
        self.data.set_spearman_correlation(self.data.scores[dataset], predicted_scores)
        self.data.print_results(self.name, self.data.token_transformer_name, dataset)


def evaluate_siamese_lstm(language: str) -> None:
    siamese_lstm = SiameseLSTM(language=language, verbose=Verbose.SILENT)
    # siamese_lstm.train(epochs=10)
    # siamese_lstm.evaluate()


def main() -> None:
    siamese_lstm = SiameseLSTM(language=parse_program_args())
    print('Embedding dim:', siamese_lstm.data.embedding_dim)
    print('Number of tokens:', siamese_lstm.data.number_of_tokens)
    # siamese_lstm.train(epochs=10)
    # siamese_lstm.evaluate(dataset='Train')
    # siamese_lstm.evaluate()


if __name__ == '__main__':
    main()
