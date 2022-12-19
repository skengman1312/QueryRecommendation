import pandas as pd
import numpy as np


class DataSet:
    """
    First class for the generation of a dataset, for now only categorical variables
    all the attributes have the same number of discrete values
    """

    def __init__(self, n_entries, n_attributes, attribute_variations):
        attr_vals = [f"value_{i}" for i in range(attribute_variations)]  # the values for all the attributes
        #p = np.random.normal(loc=5, scale=1, size=attribute_variations) # does not work yet
        print(p)
        d = {f"attr_{i}": np.random.choice(attr_vals, size=n_entries,p = p) for i in range(n_attributes)}
        df = pd.DataFrame(d)
        self.table = df


if __name__ == "__main__":
    d = DataSet(n_entries=10000, n_attributes=5, attribute_variations=6)
    print(d.table["attr_0"].value_counts())
