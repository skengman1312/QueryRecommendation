import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class DataSet:
    """
    First class for the generation of a dataset
    all the categorical attributes have the same number of discrete values
    all the continuous attributes are drawn from the same distribution
    """

    def __init__(self, n_entries=10000, n_discrete_attributes=5, n_continuous_attributes=5,
                 discrete_attribute_variations=10):

        attr_vals = [f"value_{i}" for i in range(discrete_attribute_variations)]  # the values for all the attributes
        p = np.random.normal(loc=100, scale=1, size=discrete_attribute_variations)  # ugly way to generate non uniform
        # probability vector, to be improved
        p = p / p.sum()
        d = {f"attr_{i}": np.random.choice(attr_vals, size=n_entries, p=np.random.shuffle(p)) for i in
             range(n_discrete_attributes)}
        dd = {f"attr_{i}": np.random.normal(loc=5, scale=2, size=n_entries) for i in
              range(n_discrete_attributes, n_discrete_attributes + n_continuous_attributes)}
        # lines 18-24 can be condensed in a single line of code if needed
        # TODO: add support for other distributions and/or parameter tuning
        self.table = pd.DataFrame({**d, **dd})
        self.log = None

    def query_gen(self):
        """
        generator for queries
        """
        # randomly samples from norm the number of conditions for each query
        query_len = int(np.random.normal(loc=self.table.shape[1] // 3, scale=1, size=1).clip(1, self.table.shape[1]))
        query_attr = np.random.choice(self.table.columns.values, replace=False, size=query_len)

        # choice of the value for each attribute in the condition
        disc_query_dict = {attr: np.random.choice(self.table[attr].unique()) for attr in query_attr if
                           self.table[attr].dtype == "object"}
        cont_query_dict = {attr: np.round(decimals=4, a=np.random.normal(loc=5, scale=1)) for attr in query_attr if
                           self.table[attr].dtype == "float64"}
        q = Query(0, [(attr, "==", f"'{value}'") for attr, value in disc_query_dict.items()] +
                  [(attr, np.random.choice((">", "<")), value) for attr, value in cont_query_dict.items()])

        if len(self.query(q)) > 0:  # yields only non-null queries
            yield q
        else:
            yield next(self.query_gen())

    def unique_query_log_gen(self, log_len, disable=False):
        """
        generates a log of unique queries
        """
        # we have to decide whether we make each query unique or not,
        # in general I doubt there will be many duplicates,
        # we can also consider a datastructures dedicated to query logs aas a stand-alone class
        log = list()
        with tqdm(total=log_len, desc="Query log generation", disable=disable) as pbar:
            while len(log) < log_len:
                q = next(self.query_gen())
                if q not in log:
                    q.id = len(log)
                    log.append(q)
                    pbar.update(1)
        self.log = log
        return log

    def query(self, q):
        """
        query function to call pandas query
        :param q: query to be evaluated
        :return:
        """
        return self.table.query((str(q)))

    def save_csv(self, filepath="../data/dataset.csv"):
        """
        Exports the object as a csv to the specified path
        :param filepath: path of the csv
        :return:
        """
        self.table.to_csv(filepath)

    @classmethod
    def from_csv(cls, filepath="../data/dataset.csv"):
        """
        Loads a csv file as the DataSet
        :param filepath: path of the csv file containing the dataset
        :return: DataSet obj initialized with the table stored in the csv
        """
        td = cls(0, 0, 0, 0)
        td.table = pd.read_csv(filepath, index_col=0)
        return td


class Query:
    """
    simple class to hold a single query
    """

    def __init__(self, identifier: int, conditions):
        self.id = identifier
        self.attr = [a[0] for a in conditions]  # variable to hold the attributes included in the query,
        # can speed up the comparison
        assert len(self.attr) == len(set(self.attr)), "only one condition per attribute is allowed"
        self.conditions = conditions
        self.conditions.sort()

    def __str__(self):
        """
        string representation method
        :return:
        """
        s = ["({} {} {})".format(*i) for i in self.conditions]
        return f'({" & ".join(s)})'

    def __repr__(self):
        return f"{self.id}::{self.__str__()}"  # to fix probably

    def __eq__(self, other):
        """
        Method used to ovverride the == operator and to allow faster equivalence comparison between queries
        :param other: The second Query obj in the comparison
        :return:
        """
        return set(self.attr) == set(other.attr) and self.conditions == other.conditions


class User:
    """
    A single user
    """

    def __init__(self, dataset, identifier=0):
        self.id = identifier
        self.queries = None
        self.seed = None
        self.iseed = None
        self.dataset = dataset

    def random_qseed(self, n=6):
        """
        Generates a set of seed queries used for evaluation
        :param n: number of queries included in the seed
        :return:
        """
        self.queries = self.dataset.unique_query_log_gen(log_len=n, disable=True)
        self.seed = pd.concat([self.dataset.query(q) for q in self.queries], axis=0)
        self.iseed = self.seed.index

        if len(self.iseed) < 10000:  # check that the query seeds poit at sufficiently large portion of the dataset,
            # otherwise, the user will be very difficult to satisfy
            self.random_qseed(n=n)

    def rate(self, q):
        """
        Function to produce rating of a query q
        :param q: query to be rated by the user
        :return:
        """
        # print(self.dataset.table)
        qi = self.dataset.query(q).index  # index of the values returned by the query q
        # the query is rated:
        # (number of rows pointed both by seed and rated query) / (number of rows pointed by rated query)
        return np.round(len(qi.intersection(self.iseed)) / len(qi), decimals=4) if len(qi) > 0 else 0


class UtilityMatrix:
    """
    Class that computes and stores the utility matrix, requires a dataset
    """

    def __init__(self, dataset, n_queries, n_users, n_queries_per_user):
        self.dataset = dataset
        self.queries = pd.Series(dataset.unique_query_log_gen(n_queries))
        self.users = [User(dataset, identifier=i) for i in range(n_users)]
        [u.random_qseed() for u in self.users]
        self._ratings = [[(q.id, u.rate(q))
                          for q in np.random.choice(self.queries, size=n_queries_per_user, replace=False)]
                         for u in tqdm(self.users)]
        self._ratings = [pd.DataFrame(r).set_index(0) for r in self._ratings]

        self.ratings = pd.concat(self._ratings, axis=1, ignore_index=False).sort_index()
        self.ratings.columns = list(range(len(self.users)))
        self.ratings = self.ratings.transpose()

    def export_csv(self, filepath):
        self.ratings.to_csv(f"{filepath}/utility_matrix.csv")
        self.queries.astype(str).str.replace(r"[\(*\)*]", "").str.split("&", expand=True).to_csv(
            f"{filepath}/query_log.csv")
        pd.DataFrame([u.id for u in self.users]).set_index(0).to_csv(f"{filepath}/user_list.csv")


if __name__ == "__main__":
    d = DataSet(n_entries=100000, n_discrete_attributes=5, discrete_attribute_variations=100)

    # print(d.table["attr_0"].value_counts())
    # d.table["attr_0"].value_counts().sort_index().plot()
    # d.table["attr_9"].plot(kind="hist")
    # plt.show()
    # print(d.unique_query_log_gen(5000))
    # for name in d.table.columns:
    #     print(name)
    #     print(d.table[name].dtype)
    # q = next(d.query_gen())
    # print(q)
    # print(q.conditions)
    # print(d.query(q))
    # q = Query(0, [("attr_2", "==", "'value_8'"), ("attr_1", "==", "'value_0'"), ("attr_8", "<", 6.0794)])
    # print(q)
    # print(d.query(q))
    # qq = Query(1, [("attr", "==", "val"), ("attr0", "==", "val0")])
    # l = [q]
    # print(qq in l)
    # qqq = Query(2, [("attr0", "==", "val0"), ("attr7", "==", "val2")])

    # log = d.unique_query_log_gen(1000)
    # u = User(d)
    # u.random_qseed()
    # log_rating = pd.DataFrame([u.rate(q) for q in log]).sort_values(0)
    # log_rating.plot(kind="hist", bins=100)
    # print(f"mean: {log_rating.mean()}")
    # print(f"max: {log_rating.max()}")
    # print(f"min: {log_rating.min()}")
    #
    # plt.show()
    d.save_csv()

    um = UtilityMatrix(d, 1000, 50, 600)
    um.export_csv("../data/")
