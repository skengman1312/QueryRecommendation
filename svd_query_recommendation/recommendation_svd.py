import numpy as np
from numpy.linalg import svd


def rec(fm):
    np.random.seed(1312)
    u, s, vh = np.linalg.svd(fm, full_matrices=False)
    user_0_concepts = np.where(u[0] > 0)[0]
    user_0_concepts = u[0].argpartition(-3)[-3:]
    user_O_queries = {i: vh[i].argpartition(-5)[-5:] for i in user_0_concepts}
    print(user_0_concepts)
    print(user_O_queries)
    return u, s, vh
