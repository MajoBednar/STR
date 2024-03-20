from sentence_transformers import SentenceTransformer

from program_args import parse_program_args
from load_data import load_data


def main(language: str = 'eng') -> None:
    scores, sentence_pairs = load_data(language=input('Language: '), dataset='_train.csv')
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print(sentence_pairs[:10])
    # sentences_1 = []
    # sentences_2 = []
    # for pair in sentence_pairs:
    #     sentences_1.append(pair[0])
    #     sentences_2.append(pair[1])

    # embeddings_1 = model.encode(sentences_1)
    # print(sentences_1[0], embeddings_1[0])


if __name__ == '__main__':
    main(parse_program_args())
