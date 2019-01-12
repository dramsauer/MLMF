import sklearn.metrics.pairwise as sk


def cosine_similarity(x, y):
    """
    Calculate the cosine similarity between two vectors with sklearn package.
    :param x and
    :param y the feature vectors
    :return: number between 0 and 1, with 1 as high similarity.
    """
    return sk.cosine_similarity([x], [y])
