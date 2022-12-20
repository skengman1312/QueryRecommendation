import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataSet:
    """
    First class for the generation of a dataset
    all the categorical attributes have the same number of discrete values
    all the continuous attributes are drawn from the same distribution
    """

    def __init__(self, n_entries=10000, n_discrete_attributes=10, n_continuous_attributes=0,
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
        query_len = int(np.random.normal(loc=self.table.shape[1] // 3, scale=1, size=1).clip(1, self.table.shape[1]))
        # randomly samples from norm the number of conditions for each query
        query_attr = np.random.choice(self.table.columns.values, replace=False, size=query_len)
        query_dict = {attr: np.random.choice(self.table[attr].unique()) for attr in query_attr}
        yield Query(0, [(attr, "==", value) for attr, value in query_dict.items()])
        # needs to change when we consider also continuous attributes

    def unique_query_log_gen(self, log_len):
        """
        generates a log of unique queries
        """
        # we have to decide whether or not make each query unique or not,
        # in general i doubt there will be many duplicates,
        # we can also consider a datastructures dedicated to query logs aas a stand alone class
        log = list()
        while len(log) < log_len:
            q = next(self.query_gen())
            if q not in log:
                q.id = len(log)
                log.append(q)
        self.log = log
        return log


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
        if set(self.attr) != set(other.attr):
            return False
        if self.conditions == other.conditions:
            return True
        else:
            return False


if __name__ == "__main__":
    d = DataSet(n_entries=10000, n_discrete_attributes=10, discrete_attribute_variations=100)

    # print(d.table["attr_0"].value_counts())
    # d.table["attr_0"].value_counts().sort_index().plot()
    # d.table["attr_9"].plot(kind="hist")
    # plt.show()
    print(d.unique_query_log_gen(5))
    # print(next(d.query_gen()))

    # q = Query(0, [("attr", "==", "val"), ("attr0", "==", "val0")])
    # qq = Query(1, [("attr", "==", "val"), ("attr0", "==", "val0")])
    # l = [q]
    # print(qq in l)
    # qqq = Query(2, [("attr0", "==", "val0"), ("attr", "==", "val")])
