import random
import pandas as pd

class DataSet:
    """
    First class for the generation of a dataset, for now only categorical variables
    all the attributes have the same number of discrete values
    """
    def __init__(self, n_entries, n_attributes, attribute_variations):
        attr_vals = [f"value_{i}" for i in range(attribute_variations)]  # the values for all the attributes
        d = {f"attr_{i}": random.choices(attr_vals, k= n_entries) for i in range(n_attributes)}
        df = pd.DataFrame(d)
        print(df)



if __name__ == "__main__":
            d = DataSet(n_entries=10, n_attributes= 5, attribute_variations=6)