import torch
import torch.nn as nn
from torch.optim import Adam

from src.utilities.program_args import parse_program_args
from src.utilities.constants import Verbose
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
    def __init__(self, language: str, data_split: str, transformer_name: str = 'all MiniLM', learning_rate: float = 0.001,
                 verbose: Verbose = Verbose.DEFAULT):
        super().__init__(verbose)
        self.name = 'Siamese MLP'
        self.data = DataManagerWithSentenceEmbeddings.load(language, data_split, transformer_name)

        self.model = SiameseMLPArchitecture(self.data.embedding_dim)
        self.loss_function = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

    def train(self, epochs: int = 1, batch_size: int = 32, patience: int = 20):
        best_val_correlation = -1
        best_val_loss = float('inf')
        no_improvement_count = 0
        best_model_state = None

        for epoch in range(epochs):
            running_loss = 0.0
            for batch in range(0, len(self.data.sentence_pairs['Train']), batch_size):
                self.optimizer.zero_grad()

                inputs1 = torch.tensor(self.data.sentence_embeddings['Train'][0][batch:batch + batch_size])
                inputs2 = torch.tensor(self.data.sentence_embeddings['Train'][1][batch:batch + batch_size])
                true_scores = torch.tensor(self.data.scores['Train'][batch:batch + batch_size])

                outputs = self.model(inputs1, inputs2)
                true_scores = true_scores.unsqueeze(1)
                loss = self.loss_function(outputs, true_scores)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs1.size(0)

            epoch_loss = running_loss / len(self.data.sentence_pairs['Train'])

            with torch.no_grad():
                input1 = torch.tensor(self.data.sentence_embeddings['Dev'][0])
                input2 = torch.tensor(self.data.sentence_embeddings['Dev'][1])
                true_scores_val = torch.tensor(self.data.scores['Dev'])
                true_scores_val = true_scores_val.unsqueeze(1)
                predicted_scores_val = self.model(input1, input2)
                val_loss = self.loss_function(predicted_scores_val, true_scores_val)
                val_correlation = self.data.calculate_spearman_correlation(true_scores_val, predicted_scores_val)

            if val_correlation > best_val_correlation:
                best_val_correlation = val_correlation
                no_improvement_count = 0
                best_model_state = self.model.state_dict()
            else:
                no_improvement_count += 1

            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     no_improvement_count = 0
            #     best_model_state = self.model.state_dict()
            # else:
            #     no_improvement_count += 1

            if self.verbose == Verbose.DEFAULT or self.verbose == Verbose.EXPRESSIVE:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
                print(f'Val Correlation: {val_correlation}')
            if self.verbose == Verbose.EXPRESSIVE:
                self.evaluate()

            if no_improvement_count >= patience:
                print(f'No improvement in test loss for {patience} epochs. Early stopping.')
                break

        # Restore the best model state
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def evaluate(self, dataset: str = 'Test'):
        input1 = torch.tensor(self.data.sentence_embeddings[dataset][0])
        input2 = torch.tensor(self.data.sentence_embeddings[dataset][1])
        with torch.no_grad():
            predicted_scores = self.model(input1, input2)
            true_scores = torch.tensor(self.data.scores[dataset])
            true_scores = true_scores.unsqueeze(1)

        if self.verbose == Verbose.EXPRESSIVE:
            print(predicted_scores)

        self.data.set_spearman_correlation(true_scores, predicted_scores)
        self.data.print_results(self.name, self.data.sentence_transformer_name, dataset)


def evaluate_siamese_mlp(language: str, data_split: str) -> None:
    siamese_mlp = SiameseMLP(language=language, data_split=data_split, verbose=Verbose.SILENT)
    siamese_mlp.train(epochs=10)
    siamese_mlp.evaluate()


def main() -> None:
    language, data_split = parse_program_args()
    siamese_mlp = SiameseMLP(language, data_split, transformer_name='all MiniLM')
    siamese_mlp.train(epochs=100, patience=20)
    siamese_mlp.evaluate(dataset='Train')
    siamese_mlp.evaluate()


if __name__ == '__main__':
    main()
