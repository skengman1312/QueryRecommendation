import numpy as np
import pandas as pd
from numpy.linalg import svd


class QSRS:
    """
    Query SVD Recommendation System class
    """

    def __init__(self, um):
        self._res_counts = None
        self.top_res = None
        self.top_q = None
        self.um = um
        np.random.seed(1312)
        if um.filled_matrix is None:
            um.fill()
        self.u, self.s, self.vh = np.linalg.svd(um.filled_matrix, full_matrices=False)

    def get_top_q(self, k):
        """
        Gets the top k queries for each concept and loads them
        :param k:
        """
        self.top_q = {i: self.vh[i].argpartition(-k)[-k:] for i in range(self.u.shape[0])}
        pd.concat([self.um.dataset.query(self.um.queries[q]) for q in self.top_q[10]])
        self.top_res = [pd.concat([self.um.dataset.query(self.um.queries[q]) for q in self.top_q[i]])
                        for i in range(len(self.top_q))]
        self._res_counts = {i: self.top_res[i].value_counts()[:5].reset_index() for i in range(len(self.top_q))}

    # def top_queries_per_concept(self):

    def recommendation(self, user_id, length: int):
        """

        :param length: number of queries to be recommended
        :param user_id: user to get the recommendation for
        """

        def generate_query(cdf):
            """
            count data frame
            :param cdf:
            """
            pert = 1
            while True:
                cdf[0] *= pert

                q = list()
                for attr in cdf.columns[:-1]:  # excluding the = index of the counts
                    d = pd.Series({v: cdf[0][cdf[attr] == v].sum() for v in cdf[attr].unique()})
                    if d.max() > d.sum() / 2:
                        q.append([attr, "==", f'"{d.idxmax()}"'])

                yield q
                pert = [np.random.uniform(0.2, 1.8) for _ in range(cdf.shape[0])]

        if not self._res_counts:
            self.get_top_q(5)
        top3c = self.u[user_id].argpartition(-3)[-3:]
        generators = [generate_query(self._res_counts[c]) for c in top3c]
        qq, i = list(), 0
        while len(qq) < length:
            q = next(generators[i % 3])
            if q not in qq:
                qq.append(q)
        return qq

    # TODO: implement "similarity" parameter for user


def rec(fm):
    np.random.seed(1312)
    u, s, vh = np.linalg.svd(fm, full_matrices=False)
    user_0_concepts = np.where(u[0] > 0)[0]
    user_0_concepts = u[0].argpartition(3)[3:]
    user_O_queries = {i: vh[i].argpartition(-5)[-5:] for i in user_0_concepts}
    print(user_0_concepts)
    print(user_O_queries)
    return u, s, vh
