from src.utilities.program_args import parse_program_args
from src.utilities.data_management import DataManager
from src.embeddings.token_embeddings import DataManagerWithTokenEmbeddings
from src.embeddings.sentence_embeddings import DataManagerWithSentenceEmbeddings
from src.baselines import str_lexical_overlap as lo, str_cosine_similarity as cs
from src.models import str_word_movers_distance as wmd, str_siamese_mlp as mlp, str_siamese_lstm as lstm


if __name__ == '__main__':
    language, data_split = parse_program_args()
    transformer = 'LaBSE'

    data_manager = DataManager(language, data_split)
    data_manager_token_embeddings = DataManagerWithTokenEmbeddings.load(language, data_split, transformer)
    data_manager_sentence_embeddings = DataManagerWithSentenceEmbeddings.load(language, data_split, transformer)

    lo.evaluate_lexical_overlap(data_manager)
    cs.evaluate_cosine_similarity(data_manager_sentence_embeddings)
    wmd.evaluate_word_movers_distance(data_manager_token_embeddings)

    mlp.evaluate_siamese_mlp(data_manager_sentence_embeddings)
    lstm.evaluate_siamese_lstm(data_manager_token_embeddings)
