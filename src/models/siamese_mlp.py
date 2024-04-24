import torch
import torch.nn as nn
from torch.optim import Adam

from src.utilities.program_args import parse_program_args
from src.utilities.constants import Verbose
from src.embeddings.sentence_embeddings import DataManagerWithSentenceEmbeddings


class SiameseMLPForSentenceEmbeddings(nn.Module):
    def __init__(self, input_dim):
        super(SiameseMLPForSentenceEmbeddings, self).__init__()

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
            # nn.ReLU(),
        )

    def forward(self, embedding_sentence1, embedding_sentence2):
        out1 = self.shared_branch(embedding_sentence1)
        out2 = self.shared_branch(embedding_sentence2)

        combined = torch.abs(out1 - out2)
        out = torch.sigmoid(self.common_branch(combined))
        return out


class SiameseMLP:
    def __init__(self, language: str, learning_rate: float = 0.001, verbose: Verbose = Verbose.DEFAULT):
        self.name = 'Siamese Network for Sentence Embeddings'
        self.data = DataManagerWithSentenceEmbeddings.load(language)
        self.verbose: Verbose = verbose

        self.model = SiameseMLPForSentenceEmbeddings(self.data.embedding_dim)
        self.loss_function = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

    def train(self, epochs: int = 1, batch_size: int = 32):
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

            input1 = torch.tensor(self.data.sentence_embeddings['Test'][0])
            input2 = torch.tensor(self.data.sentence_embeddings['Test'][1])
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
        input1 = torch.tensor(self.data.sentence_embeddings[dataset][0])
        input2 = torch.tensor(self.data.sentence_embeddings[dataset][1])
        with torch.no_grad():
            predicted_scores = self.model(input1, input2)

        if self.verbose == Verbose.EXPRESSIVE:
            print(predicted_scores)
        self.data.calculate_spearman_correlation(self.data.scores[dataset], predicted_scores)
        self.data.print_results(self.name, dataset)


def evaluate_siamese_sentence(language: str) -> None:
    siamese_sentence = SiameseMLP(language=language, verbose=Verbose.SILENT)
    siamese_sentence.train(epochs=10)
    siamese_sentence.evaluate()


def main() -> None:
    siamese_sentence = SiameseMLP(language=parse_program_args())
    siamese_sentence.train(epochs=10)
    siamese_sentence.evaluate(dataset='Train')
    siamese_sentence.evaluate()


if __name__ == '__main__':
    main()
