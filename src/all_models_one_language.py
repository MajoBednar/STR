from src.utilities.program_args import parse_program_args
from src.baselines import lexical_overlap as lo, cosine_similarity as cs, linear_regression as lr
from src.models import siamese_mlp as mlp


if __name__ == '__main__':
    language, data_split = parse_program_args()
    transformer = 'LaBSE'

    lo.evaluate_lexical_overlap(language=language, data_split=data_split)
    cs.evaluate_cosine_similarity(language=language, data_split=data_split, transformer_name=transformer)
    # lr.evaluate_linear_regression(language=language, data_split=data_split)

    mlp.evaluate_siamese_mlp(language=language, data_split=data_split, transformer_name=transformer)
