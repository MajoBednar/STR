import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import paired_cosine_distances


def cosine_similarity(vector1, vector2) -> float:
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = norm(vector1)
    norm_vector2 = norm(vector2)
    return dot_product / (norm_vector1 * norm_vector2)


def cosine_similarities(vectors1, vectors2):
    cosine_scores = 1 - paired_cosine_distances(vectors1, vectors2)
    cosine_scores = cosine_scores.flatten().tolist()
    return cosine_scores
