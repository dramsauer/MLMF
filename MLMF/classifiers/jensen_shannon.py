import scipy.special

import scipy

def jensen_shannon_divergence(p, q):
    """
    Calculating the Jensen-Shannon-Divergence, based on this stackoverflow solution
    https://stackoverflow.com/questions/15880133/jensen-shannon-divergence
    scipy.stats.entropy(p,q) is the Kullback-Leibler-Divergence of two probability distributions p and q
    :param p: and
    :param q: the two probability distributions
    :return: the Jensen-Shannon-Divergence
    """
    p_normed = p / scipy.linalg.norm(p, ord=1)
    q_normed = q / scipy.linalg.norm(q, ord=1)
    M = 0.5 * (p_normed + q_normed)
    return 0.5 * (scipy.stats.entropy(p_normed, M) + scipy.stats.entropy(q_normed, M))
