# import datagen.generator as gen
# import datagen.utility_svd as svd

from datagen.generator import *
from datagen.utility_svd import *

d = DataSet(n_entries=10000, n_discrete_attributes=5, discrete_attribute_variations=100)
um = UtilityMatrix(d, 100, 10, 60)


def ctest(um, usid, qid):
    fm, _ = SVT(um.ratings, max_iter=1000)
    fm = pd.DataFrame(fm)
    print(um.queries[qid])
    print(f"truth: {um.users[usid].rate(um.queries[qid])}")
    print(fm.iloc[usid, qid])


def full_matrix_test(um):
    for c in um.ratings.columns:
        print(um.ratings.isna())
        break


