from src.utilities.program_args import parse_program_args
from src.baselines import lexical_overlap as lo, cosine_similarity as cs, linear_regression as lr
from src.models import siamese_mlp as mlp


if __name__ == '__main__':
    language = parse_program_args()

    lo.evaluate_lexical_overlap(language=language)
    cs.evaluate_cosine_similarity(language=language)
    lr.evaluate_linear_regression(language=language)

    mlp.evaluate_siamese_mlp(language=language)
