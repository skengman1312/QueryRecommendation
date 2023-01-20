import numpy as np
from numpy.linalg import svd


class recco:

    def __init__(self, um):
        self.top_q = None
        self.um = um
        np.random.seed(1312)
        if um.filled_matrix is None:
            um.fill()
        self.u, self.s, self.vh = np.linalg.svd(um.filled_matrix, full_matrices=False)


    def get_top_q(self, k):

        self.top_q = {i: self.vh[i].argpartition(-k)[-k:] for i in range(self.u.shape[0])}

    # def top_queries_per_concept(self):


def rec(fm):
    np.random.seed(1312)
    u, s, vh = np.linalg.svd(fm, full_matrices=False)
    user_0_concepts = np.where(u[0] > 0)[0]
    user_0_concepts = u[0].argpartition(-3)[-3:]
    user_O_queries = {i: vh[i].argpartition(-5)[-5:] for i in user_0_concepts}
    print(user_0_concepts)
    print(user_O_queries)
    return u, s, vh
