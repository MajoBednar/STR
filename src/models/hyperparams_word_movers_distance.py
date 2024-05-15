from src.utilities.program_args import parse_program_args
from src.utilities.constants import TOKEN_TRANSFORMERS
from src.utilities.hyperparameter_tuning import find_best_transformers
from src.embeddings.token_embeddings import DataManagerWithTokenEmbeddings
from .str_word_movers_distance import STRWordMoversDistance


def main() -> None:
    language, data_split = parse_program_args()
    best_transformers = {'Train': '', 'Dev': '', 'Test': ''}
    best_correlations = {'Train': -1.0, 'Dev': -1.0, 'Test': -1.0}
    # the only hyperparameter for the STR word movers distance model is a token transformer
    for transformer in TOKEN_TRANSFORMERS:
        data_manager = DataManagerWithTokenEmbeddings.load(language, data_split, transformer, False)
        model = STRWordMoversDistance(data_manager)
        best_transformers, best_correlations = find_best_transformers(model, transformer, best_transformers,
                                                                      best_correlations)

    print(best_transformers)
    print(best_correlations)


if __name__ == '__main__':
    main()
