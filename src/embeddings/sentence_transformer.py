from sentence_transformers import SentenceTransformer


def create_sentence_embeddings(pair_of_sentences: tuple[list[str], list[str]]) -> tuple[list, list]:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_embeddings1 = model.encode(pair_of_sentences[0])
    sentence_embeddings2 = model.encode(pair_of_sentences[1])
    return sentence_embeddings1, sentence_embeddings2
