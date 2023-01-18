# import datagen.generator as gen
# import datagen.utility_svd as svd
import pandas as pd
from tqdm import tqdm

from datagen.generator import *
from datagen.utility_svd import *


def ctest(um, usid, qid):
    fm, _ = SVT(um.ratings, max_iter=1000)
    fm = pd.DataFrame(fm)
    print(um.queries[qid])
    print(f"truth: {um.users[usid].rate(um.queries[qid])}")
    print(fm.iloc[usid, qid])


def full_matrix_test(um):
    """
    takes as input a utility matrix and calculates the RMSE of the predictions
    :param um: Utility Matrix
    :return: RMSE
    """
    fm, _ = SVT(um.ratings, max_iter=1500)
    fm = fm.clip(0,1)
    fm = pd.DataFrame(fm)

    s, c = 0, 0


    for i in tqdm(um.ratings, desc="RMSE calculation", total=len(um.ratings.columns)):
        index = um.ratings[i][um.ratings[i].isna()].index
        # print(um.ratings[i][um.ratings[i].isna()])
        gt = [um.users[uid].rate(um.queries[i]) for uid in index]
        pred = fm[i][index]
        s_err = (gt - pred) ** 2
        s += s_err.sum()
        c += len(s_err)
        # print(f"ground truth :{gt}")
        # print(f"prediction: {pred}")
        # print(gt-pred)
        # print(f"error: {s_err.sum()}")
        # print(c)
    print(s)
    print(c)
        # break
    return (s / c) ** (1/2)


if __name__ == "__main__":
    d = DataSet(n_entries=100000, n_discrete_attributes=5, n_continuous_attributes=2, discrete_attribute_variations=100)
    um = UtilityMatrix(d, 1000, 10, 800)
    fm, _ = SVT(um.ratings, max_iter=1500)
    print(fm)

    print(full_matrix_test(um))
