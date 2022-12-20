import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataSet:
    """
    First class for the generation of a dataset, for now only categorical variables
    all the attributes have the same number of discrete values
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
        # TODO: add support for other distributions and/or parameter tuning
        df = pd.DataFrame({**d, **dd})
        self.table = df


if __name__ == "__main__":
    d = DataSet(n_entries=10000, n_discrete_attributes=5, discrete_attribute_variations=100)
    # print(d.table["attr_0"].value_counts())
    # d.table["attr_0"].value_counts().sort_index().plot()
    d.table["attr_9"].plot(kind="hist")
    plt.show()
    print(d.table)
