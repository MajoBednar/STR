import torch
import torch.nn as nn
from torch.optim import Adam

from src.utilities.program_args import parse_program_args
from src.embeddings.sentence_embeddings import DataManagerWithSentenceEmbeddings


class SiameseNetworkForSentences(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetworkForSentences, self).__init__()

        # Shared branch of the Siamese network
        self.shared_branch = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # Additional layers for combining outputs (optional)
        self.fc = nn.Linear(16, 1)
        nn.ReLU()

    def forward(self, x1, x2):
        # Process each sentence embedding through the shared branch
        out1 = self.shared_branch(x1)
        out2 = self.shared_branch(x2)

        # Combine the outputs of the shared branches
        combined = torch.abs(out1 - out2)  # Example of element-wise subtraction

        # Additional layers for combining outputs (optional)
        combined = self.fc(combined)

        return combined
    # def __init__(self, input_dim):
    #     super(SiameseNetworkForSentences, self).__init__()
    #
    #     self.shared_branch = nn.Sequential(
    #         nn.Linear(input_dim, 512),
    #         nn.ReLU(),
    #         # nn.Linear(512, 256),
    #         # nn.ReLU(),
    #         # nn.Linear(256, 128),
    #         # nn.ReLU()
    #         nn.Linear(512, 128),
    #         nn.ReLU()
    #     )
    #
    #     self.common_branch = nn.Sequential(
    #         nn.Linear(128, 32),
    #         nn.ReLU(),
    #         nn.Linear(32, 1),
    #         nn.ReLU(),
    #     )
    #
    # def forward(self, embedding_sentence1, embedding_sentence2):
    #     out1 = self.shared_branch(embedding_sentence1)
    #     out2 = self.shared_branch(embedding_sentence2)
    #
    #     combined = out1 + out2
    #     out = self.common_branch(combined)
    #     return out


class SiameseSentence:
    def __init__(self, language: str, learning_rate: float = 0.001):
        self.name = 'Siamese Network for Sentence Embeddings'
        self.data = DataManagerWithSentenceEmbeddings(language)
        self.model = SiameseNetworkForSentences(self.data.embedding_dim)
        self.loss_function = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

    def train(self, epochs: int = 1, batch_size: int = 32):
        for epoch in range(epochs):
            running_loss = 0.0
            for batch in range(0, len(self.data.sentence_pairs_train), batch_size):
                self.optimizer.zero_grad()

                inputs1 = torch.tensor(self.data.sentence_embeddings_train[0][batch:batch + batch_size])
                inputs2 = torch.tensor(self.data.sentence_embeddings_train[1][batch:batch + batch_size])
                true_scores = torch.tensor(self.data.scores_train[batch:batch + batch_size])

                outputs = self.model(inputs1, inputs2)
                true_scores = true_scores.unsqueeze(1)
                loss = self.loss_function(outputs, true_scores)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs1.size(0)

            epoch_loss = running_loss / len(self.data.sentence_pairs_train)

            input1 = torch.tensor(self.data.sentence_embeddings_test[0])
            input2 = torch.tensor(self.data.sentence_embeddings_test[1])
            true_scores_dev = torch.tensor(self.data.scores_train)
            with torch.no_grad():
                predicted_scores_dev = self.model(input1, input2)
                dev_loss = self.loss_function(predicted_scores_dev, true_scores_dev)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {dev_loss}")

    def evaluate(self):
        predicted_scores = []
        for i in range(len(self.data.sentence_pairs_test)):
            input1 = torch.tensor(self.data.sentence_embeddings_test[0][i])
            input2 = torch.tensor(self.data.sentence_embeddings_test[1][i])
            with torch.no_grad():
                predicted_scores.append(self.model(input1, input2).item())

        # print(predicted_scores)
        self.data.calculate_spearman_correlation(self.data.scores_test, predicted_scores)
        self.data.print_results(self.name)


if __name__ == '__main__':
    siamese_sentence = SiameseSentence(language=parse_program_args())
    siamese_sentence.train(epochs=15)
    siamese_sentence.evaluate()
