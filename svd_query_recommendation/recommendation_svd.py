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
        # pd.concat([self.um.dataset.query(self.um.queries[q]) for q in self.top_q[10]])
        self.top_res = [pd.concat([self.um.dataset.query(self.um.queries[q]) for q in self.top_q[i]])
                        for i in range(len(self.top_q))]
        self._res_counts = {i: self.top_res[i].value_counts()[:5].reset_index() for i in range(len(self.top_q))}

    # def top_queries_per_concept(self):

    def recommendation(self, user_id, length: int):
        """
        Recommend taking into account the most associated queries for concepts mostly associated to the user
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
            i += 1
        return qq

    def recommendationV2(self, user_id, length: int):
        """
        Recommend taking into account the top queries as rated by the user for concepts mostly associated to the user
        count variant
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
            self.get_top_q(10)
        top3c = self.u[user_id].argpartition(-3)[-3:]

        # r.um.filled_matrix.loc[10][r.top_q[11]] get rating for the top queries of concept 11 by user 10
        # r.um.filled_matrix.loc[10][r.top_q[11]].sort_values(ascending=False)[:5].reset_index()["index"]
        # top 5 queries of concept 11 as rated by user 10
        top_queries_per_concept = {c: self.um.filled_matrix.loc[user_id][self.top_q[c]].sort_values(
            ascending=False)[:5].reset_index()["index"] for c in top3c}
        top_res = {c: pd.concat([self.um.dataset.query(self.um.queries[q]) for q in top_queries_per_concept[c]])
                   for c in top_queries_per_concept}

        res_counts = {i: top_res[i].value_counts()[:5].reset_index() for i in top_queries_per_concept}

        # print(top_res)
        generators = [generate_query(res_counts[c]) for c in top3c]

        qq, i = list(), 0
        while len(qq) < length:
            q = next(generators[i % 3])
            if q not in qq:
                qq.append(q)
            i += 1
        return qq

    def recommendationV3(self, user_id, length: int):
        """
        Recommend taking into account the top queries as rated by the user for concepts mostly associated to the user
        weighted count variant
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
                cdf["wu"] *= pert

                q = list()
                for attr in cdf.columns[:-1]:  # excluding the = index of the counts
                    d = pd.Series({v: cdf["wu"][cdf[attr] == v].sum() for v in cdf[attr].unique()})
                    if d.max() > d.sum() / 2:
                        q.append([attr, "==", f'"{d.idxmax()}"'])

                yield q
                pert = [np.random.uniform(0.2, 1.8) for _ in range(cdf.shape[0])]

        if not self._res_counts:
            self.get_top_q(10)
        top3c = self.u[user_id].argpartition(-3)[-3:]

        # r.um.filled_matrix.loc[10][r.top_q[11]] get rating for the top queries of concept 11 by user 10
        # r.um.filled_matrix.loc[10][r.top_q[11]].sort_values(ascending=False)[:5].reset_index()["index"]
        # top 5 queries of concept 11 as rated by user 10
        top_queries_per_concept = {c: self.um.filled_matrix.loc[user_id][self.top_q[c]].sort_values(
            ascending=False)[:5].reset_index()["index"] for c in top3c}

        # top_res = {c: pd.concat([self.um.dataset.query(self.um.queries[q]) for q in top_queries_per_concept[c]])
        #            for c in top_queries_per_concept}

        top_resw = dict()  # self.um.filled_matrix[q][user_id].insert(0, "rating", self.um.filled_matrix[q][user_id])
        for c, i in top_queries_per_concept.items():
            rl = list()
            for q in i:
                r = self.um.dataset.query(self.um.queries[q])
                r.insert(len(r.columns), "rating", self.um.filled_matrix[q][user_id])
                rl.append(r)
            top_resw[c] = pd.concat(rl)

        # res_counts = {i: top_res[i].value_counts()[:5].reset_index() for i in top_queries_per_concept}

        res_countsw = {i: top_resw[i].value_counts()[:15].reset_index() for i in top_queries_per_concept}
        # print(res_countsw)
        for i, v in res_countsw.items():
            res_countsw[i].insert(len(res_countsw[i].columns), "wu", v["rating"] * v[0])
            res_countsw[i] = res_countsw[i].groupby([attr for attr in v.columns[:-3]]).mean()
            res_countsw[i] = res_countsw[i].sort_values(by="wu", ascending=False).drop(["rating", 0], axis=1).reset_index()
        # print(res_countsw)
        generators = [generate_query(res_countsw[c]) for c in top3c]

        qq, i = list(), 0
        while len(qq) < length:
            q = next(generators[i % 3])
            if q not in qq:
                qq.append(q)
            i += 1
        return qq

    def recommendationV4(self, user_id, length: int):
        """
        Recommend taking into account the top queries as rated by the user for concepts mostly associated to the user

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
                cdf["rating"] *= pert

                q = list()
                for attr in cdf.columns[:-1]:  # excluding the = index of the counts
                    d = pd.Series({v: cdf["rating"][cdf[attr] == v].sum() for v in cdf[attr].unique()})
                    if d.max() > d.sum() / 2:
                        q.append([attr, "==", f'"{d.idxmax()}"'])

                yield q
                pert = [np.random.uniform(0.2, 1.8) for _ in range(cdf.shape[0])]

        if not self._res_counts:
            self.get_top_q(10)
        top3c = self.u[user_id].argpartition(-3)[-3:]

        # r.um.filled_matrix.loc[10][r.top_q[11]] get rating for the top queries of concept 11 by user 10
        # r.um.filled_matrix.loc[10][r.top_q[11]].sort_values(ascending=False)[:5].reset_index()["index"]
        # top 5 queries of concept 11 as rated by user 10
        top_queries_per_concept = {c: self.um.filled_matrix.loc[user_id][self.top_q[c]].sort_values(
            ascending=False)[:5].reset_index()["index"] for c in top3c}

        # top_res = {c: pd.concat([self.um.dataset.query(self.um.queries[q]) for q in top_queries_per_concept[c]])
        #            for c in top_queries_per_concept}

        top_resw = dict()
        for c, i in top_queries_per_concept.items():
            rl = list()
            for q in i:
                r = self.um.dataset.query(self.um.queries[q])
                r.insert(len(r.columns), "rating", self.um.filled_matrix[q][user_id])
                rl.append(r)
            top_resw[c] = pd.concat(rl)

        # res_counts = {i: top_res[i].value_counts()[:5].reset_index() for i in top_queries_per_concept}

        res_countsw = {i: top_resw[i].value_counts().reset_index() for i in top_queries_per_concept}

        for i, v in res_countsw.items():
            # res_countsw[i].insert(len(res_countsw[i].columns), "wu", v["rating"] * v[0])
            res_countsw[i] = res_countsw[i].groupby([attr for attr in v.columns[:-3]]).mean()
            res_countsw[i] = res_countsw[i].sort_values(by="rating", ascending=False).drop([0], axis=1).reset_index()[:15]

        generators = [generate_query(res_countsw[c]) for c in top3c]

        qq, i = list(), 0
        while len(qq) < length:
            q = next(generators[i % 3])
            if q not in qq:
                qq.append(q)
            i += 1
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
