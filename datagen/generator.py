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
        generator for queries, atm working only with categorical data and equalities
        """
        # randomly samples from norm the number of conditions for each query
        query_len = int(np.random.normal(loc=self.table.shape[1] // 3, scale=1, size=1).clip(1, self.table.shape[1]))
        query_attr = np.random.choice(self.table.columns.values, replace=False, size=query_len)

        # choice of the value for each attribute in the condition
        disc_query_dict = {attr: np.random.choice(self.table[attr].unique()) for attr in query_attr if
                           self.table[attr].dtype == "object"}
        cont_query_dict = {attr: np.round(decimals=4, a=np.random.normal(loc=5, scale=1)) for attr in query_attr if
                           self.table[attr].dtype == "float64"}

        yield Query(0, [(attr, "==", f"'{value}'") for attr, value in disc_query_dict.items()] +
                    [(attr, np.random.choice((">", "<")), value) for attr, value in cont_query_dict.items()])

    def unique_query_log_gen(self, log_len):
        """
        generates a log of unique queries
        """
        # we have to decide whether or not make each query unique or not,
        # in general i doubt there will be many duplicates,
        # we can also consider a datastructures dedicated to query logs aas a stand alone class
        log = list()
        with tqdm(total=log_len, desc="Query log generation") as pbar:
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
        :param q:
        :return:
        """
        return self.table.query((str(q)))


class Query:
    """
    simple class to hold a single query, atm only equalities are accepted
    """

    def __init__(self, identifier: int, conditions):
        self.id = identifier
        self.attr = [a[0] for a in conditions]
        assert len(self.attr) == len(set(self.attr)), "only one condition per attribute is allowed"
        self.conditions = conditions
        self.conditions.sort()
        pass

    def __str__(self):
        s = ["({} {} {})".format(*i) for i in self.conditions]
        return f'({" & ".join(s)})'

    def __repr__(self):
        return f"{self.id}::{self.__str__()}"  # to fix probably

    def __eq__(self, other):
        return set(self.attr) == set(other.attr) and self.conditions == other.conditions


if __name__ == "__main__":
    d = DataSet(n_entries=1000000, n_discrete_attributes=5, discrete_attribute_variations=100)

    # print(d.table["attr_0"].value_counts())
    # d.table["attr_0"].value_counts().sort_index().plot()
    # d.table["attr_9"].plot(kind="hist")
    # plt.show()
    #print(d.unique_query_log_gen(5000))
    # for name in d.table.columns:
    #     print(name)
    #     print(d.table[name].dtype)
    q =next(d.query_gen())
    print(q)
    print(d.query(q))
    # q = Query(0, [("attr", "==", "val"), ("attr0", "==", "val0")])
    # qq = Query(1, [("attr", "==", "val"), ("attr0", "==", "val0")])
    # l = [q]
    # print(qq in l)
    # qqq = Query(2, [("attr0", "==", "val0"), ("attr", "==", "val")])
