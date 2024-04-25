from src.utilities.program_args import parse_program_args
from src.utilities.constants import Verbose
from src.embeddings.token_embeddings import DataManagerWithTokenEmbeddings


class SiameseLSTMArchitecture:
    pass


class SiameseLSTM:
    def __init__(self, language: str, learning_rate: float = 0.001, verbose: Verbose = Verbose.DEFAULT):
        self.name = 'Siamese Network for Sentence Embeddings'
        self.data = DataManagerWithTokenEmbeddings.load(language)
        self.verbose: Verbose = verbose


def evaluate_siamese_lstm(language: str) -> None:
    siamese_lstm = SiameseLSTM(language=language, verbose=Verbose.SILENT)
    # siamese_lstm.train(epochs=10)
    # siamese_lstm.evaluate()


def main() -> None:
    siamese_lstm = SiameseLSTM(language=parse_program_args())
    print('Embedding dim:', siamese_lstm.data.embedding_dim)
    print('Number of tokens:', siamese_lstm.data.number_of_tokens)
    # siamese_lstm.train(epochs=10)
    # siamese_lstm.evaluate(dataset='Train')
    # siamese_lstm.evaluate()


if __name__ == '__main__':
    main()
