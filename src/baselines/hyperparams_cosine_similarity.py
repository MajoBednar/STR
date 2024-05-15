from src.utilities.program_args import parse_program_args
from src.utilities.constants import SENTENCE_TRANSFORMERS
from src.utilities.hyperparameter_tuning import find_best_transformers
from src.embeddings.sentence_embeddings import DataManagerWithSentenceEmbeddings
from .str_cosine_similarity import STRCosineSimilarity


def main() -> None:
    language, data_split = parse_program_args()
    best_transformers = {'Train': '', 'Dev': '', 'Test': ''}
    best_correlations = {'Train': -1.0, 'Dev': -1.0, 'Test': -1.0}
    # the only hyperparameter for the STR cosine similarity model is a sentence transformer
    for transformer in SENTENCE_TRANSFORMERS:
        data_manager = DataManagerWithSentenceEmbeddings.load(language, data_split, transformer, False)
        model = STRCosineSimilarity(data_manager)
        best_transformers, best_correlations = find_best_transformers(model, transformer, best_transformers,
                                                                      best_correlations)

    print(best_transformers)
    print(best_correlations)


if __name__ == '__main__':
    main()
