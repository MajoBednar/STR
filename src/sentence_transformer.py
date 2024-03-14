from sentence_transformers import SentenceTransformer
from load_data import load_data


def main():
    scores, sentence_pairs = load_data(language=input('Language: '), dataset='_train.csv')
    model = SentenceTransformer('all-MiniLM-L6-v2')

    sentences_1 = []
    sentences_2 = []
    for pair in sentence_pairs[0]:
        sentences_1.append(pair[0])
        sentences_2.append(pair[1])

    embeddings_1 = model.encode(sentences_1)
    print(sentences_1[0], embeddings_1[0])


if __name__ == '__main__':
    main()
