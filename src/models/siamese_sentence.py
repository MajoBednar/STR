import torch.nn as nn
from sentence_transformers import SentenceTransformer

from src.utilities.program_args import parse_program_args
from src.utilities.data_management import DataManager


class SiameseNetworkForSentences(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetworkForSentences, self).__init__()

        self.shared_branch = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.common_branch = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, embedding_sentence1, embedding_sentence2):
        out1 = self.shared_branch(embedding_sentence1)
        out2 = self.shared_branch(embedding_sentence2)

        combined = out1 + out2
        out = self.common_branch(combined)
        return out


class SiameseSentence:
    def __init__(self, language: str):
        self.name = 'Siamese Network For Sentence Embeddings'
        self.data = DataManager(language)
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.model = None

    def train(self):
        pass

    def evaluate(self):
        pass


if __name__ == '__main__':
    siamese_sentence = SiameseSentence(language=parse_program_args())
    siamese_sentence.train()
    siamese_sentence.evaluate()
