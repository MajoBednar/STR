import torch
import torch.nn.functional as functional
from scipy.optimize import linear_sum_assignment

from src.embeddings.token_embeddings import DataManagerWithTokenEmbeddings
from src.utilities.program_args import parse_program_args


def solve_assignment(dist_matrix):
    # Convert distances to numpy array
    dist_matrix_np = dist_matrix.cpu().numpy()

    # Use the Hungarian algorithm to solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(dist_matrix_np)

    # Create assignment matrix
    assignment = torch.zeros_like(dist_matrix)
    assignment[row_ind, col_ind] = 1

    return assignment


def wmd(sentence1_emb, sentence2_emb):
    # Calculate pairwise distances between all embeddings
    dist_matrix = functional.pairwise_distance(sentence1_emb.unsqueeze(1), sentence2_emb.unsqueeze(0), p=2)

    # Solve the assignment problem
    # (You may need to use a more sophisticated solver for large matrices)
    assignment = solve_assignment(dist_matrix)

    # Calculate WMD
    wmd_score = (dist_matrix * assignment).sum()
    return wmd_score


def normalize_wmd(wmd_score, min_wmd, max_wmd):
    """
    Normalize WMD score to range [0, 1].

    Args:
    - wmd_score: Original WMD score
    - min_wmd: Minimum observed WMD score in the dataset
    - max_wmd: Maximum observed WMD score in the dataset

    Returns:
    - Normalized WMD score
    """
    normalized_score = (max_wmd - wmd_score) / (max_wmd - min_wmd)
    return normalized_score


class STRWordMoversDistance:
    def __init__(self, language: str, data_split: str, transformer_name: str):
        self.name = 'Word Mover\'s Distance'
        self.data = DataManagerWithTokenEmbeddings.load(language, data_split, transformer_name)

    def evaluate(self, dataset: str = 'Test') -> None:
        wmd_scores = []
        for i in range(len(self.data.scores[dataset])):
            wmd_score = wmd(self.data.token_embeddings[dataset][0][i], self.data.token_embeddings[dataset][1][i])
            wmd_scores.append(wmd_score.item())

        min_wmd, max_wmd = min(wmd_scores), max(wmd_scores)
        normalized_scores = []
        for score in wmd_scores:
            normalized_scores.append(normalize_wmd(score, min_wmd, max_wmd))

        self.data.set_spearman_correlation(self.data.scores[dataset], normalized_scores)
        self.data.print_results(self.name, self.data.transformer_name, dataset)


def evaluate_word_movers_distance(language: str, data_split: str, transformer_name: str) -> None:
    wmd_model = STRWordMoversDistance(language=language, data_split=data_split, transformer_name=transformer_name)
    wmd_model.evaluate('Test')


def main() -> None:
    language, data_split = parse_program_args()
    wmd_model = STRWordMoversDistance(language=language, data_split=data_split, transformer_name='LaBSE')
    wmd_model.evaluate('Train')
    wmd_model.evaluate('Dev')
    wmd_model.evaluate()


if __name__ == '__main__':
    main()
