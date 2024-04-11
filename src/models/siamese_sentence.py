import torch.nn as nn


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
    pass
