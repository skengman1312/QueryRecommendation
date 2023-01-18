import numpy as np
from numpy.linalg import svd


def rec(fm):
    u, s, vh = np.linalg.svd(fm, full_matrices=False)

    user_0_concepts = np.where(u[0] > 0)[0]
    user_O_queries = {i: vh[i].argpartition(5)[:5] for i in user_0_concepts}
