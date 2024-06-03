import torch
import torch.nn.functional as functional
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from pyemd import emd
import numpy as np
from collections import Counter

from src.embeddings.token_embeddings import DataManagerWithTokenEmbeddings
from src.utilities.program_args import parse_program_args


def solve_assignment(dist_matrix: torch.Tensor) -> torch.Tensor:
    dist_matrix_np = dist_matrix.cpu().numpy()  # convert distances to numpy array
    # use the Hungarian algorithm to solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(dist_matrix_np)
    # create and return assignment matrix
    assignment = torch.zeros_like(dist_matrix)
    assignment[row_ind, col_ind] = 1
    return assignment


def wmd(sentence1_emb: torch.Tensor, sentence2_emb: torch.Tensor) -> float:
    # calculate pairwise distances between all embeddings
    dist_matrix = functional.pairwise_distance(sentence1_emb.unsqueeze(1), sentence2_emb.unsqueeze(0), p=2)
    # solve the assignment problem
    assignment = solve_assignment(dist_matrix)
    # calculate and return WMD score
    return (dist_matrix * assignment).sum().item()


def create_bow(sentence: str, vocabulary: list[str]) -> np.ndarray:
    vocab_size = len(vocabulary)
    word_count = Counter(sentence.split())
    bow = np.zeros(vocab_size)
    for i, word in enumerate(vocabulary):
        bow[i] = word_count[word]
    return bow


def emd_wmd(sentence1_emb: torch.Tensor, sentence2_emb: torch.Tensor, sentence1: str, sentence2: str) -> float:
    # Compute pairwise distance matrix (Euclidean distance)
    distance_matrix = cdist(sentence1_emb, sentence2_emb, metric='euclidean')

    # BoW representations
    vocabulary = list(set(sentence1.split() + sentence2.split()))
    vocabulary.sort()
    bow_sentence_1 = create_bow(sentence1, vocabulary)
    bow_sentence_2 = create_bow(sentence2, vocabulary)
    # Normalize the BoW vectors to ensure they sum to 1
    bow_sentence_1 = bow_sentence_1 / bow_sentence_1.sum()
    bow_sentence_2 = bow_sentence_2 / bow_sentence_2.sum()

    # Compute WMD using pyemd
    return emd(bow_sentence_1, bow_sentence_2, distance_matrix)


def normalize_wmd(wmd_score: float, min_wmd: float, max_wmd: float) -> float:
    # normalize WMD score to range [0, 1]
    return (max_wmd - wmd_score) / (max_wmd - min_wmd)


class STRWordMoversDistance:
    def __init__(self, data_manager: DataManagerWithTokenEmbeddings):
        self.name: str = 'Word Mover\'s Distance'
        self.data: DataManagerWithTokenEmbeddings = data_manager

    def evaluate(self, dataset: str = 'Test') -> float:
        wmd_scores = []
        for i in range(len(self.data.scores[dataset])):
            wmd_score = wmd(self.data.token_embeddings[dataset][0][i], self.data.token_embeddings[dataset][1][i])
            wmd_score = emd_wmd(self.data.token_embeddings[dataset][0][i], self.data.token_embeddings[dataset][1][i],
                                self.data.sentence_pairs[dataset][i][0], self.data.sentence_pairs[dataset][i][1])
            wmd_scores.append(wmd_score)

        min_wmd, max_wmd = min(wmd_scores), max(wmd_scores)
        normalized_scores = []
        for score in wmd_scores:
            normalized_scores.append(normalize_wmd(score, min_wmd, max_wmd))

        correlation = self.data.calculate_spearman_correlation(self.data.scores[dataset], normalized_scores)
        self.data.print_results(correlation, self.name, self.data.transformer_name, dataset)
        return correlation


def evaluate_word_movers_distance(data_manager: DataManagerWithTokenEmbeddings) -> None:
    wmd_model = STRWordMoversDistance(data_manager)
    wmd_model.evaluate('Test')


def main() -> None:
    language, data_split = parse_program_args()
    data_manager = DataManagerWithTokenEmbeddings.load(language, data_split, 'LaBSE')

    wmd_model = STRWordMoversDistance(data_manager)
    wmd_model.evaluate('Train')
    wmd_model.evaluate('Dev')
    wmd_model.evaluate()


if __name__ == '__main__':
    main()
