from svd_query_recommendation import *


def ctest(um, usid, qid):
    """

    :param um:
    :param usid:
    :param qid:
    """
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
    if um.filled_matrix is None:
        fm, _ = SVT(um.ratings, max_iter=1500)
        fm = fm.clip(0, 1)
        fm = pd.DataFrame(fm)
    else:
        fm = um.filled_matrix

    s, c = 0, 0

    for i in tqdm(um.ratings, desc="RMSE calculation", total=len(um.ratings.columns)):
        index = um.ratings[i][um.ratings[i].isna()].index
        gt = [um.users[uid].rate(um.queries[i]) for uid in index]
        pred = fm[i][index]
        s_err = (gt - pred) ** 2
        s += s_err.sum()
        c += len(s_err)
    print(s)
    print(c)
    return (s / c) ** (1 / 2)


def test_recco(rec, um):
    recommenders = [rec.recommendationV4, rec.recommendationV3, rec.recommendationV2, rec.recommendation]
    res = list()
    mean_utility = list()
    print(um.dataset.table.shape)
    # um.dataset.table = um.dataset.table.drop_duplicates()
    print(um.dataset.table.shape)
    for recommender in recommenders:
        print(f"Testing recommender {str(recommender).split()[2]}\n")
        r = [recommender(i, 5) for i in tqdm(range(len(um.users)), desc="Generating recommended queries")]
        rr = {u.id: [u.rate(Query(0, q)) for q in r[u.id]] for u in tqdm(um.users, desc="Rating the new queries")}
        rrm = sum([sum(r) for r in rr.values()]) / (len(rr) * 5)
        res.append(rr)
        mean_utility.append(rrm)
    print(mean_utility)
    return res, mean_utility


if __name__ == "__main__":
    # Initialize the dataset from scratch and save, parent directory data must be existing

    # d = DataSet(n_entries=100000, n_discrete_attributes=5, n_continuous_attributes=0, discrete_attribute_variations=8)
    # d.save_csv("./data/dataset.csv")

    # Load the data in the Utility Matrix class and save it to directory

    # um = UtilityMatrix(d, 2000, 300, 100)
    # um.fill() # embedded method to call for the SVT algorithm
    # um.export_csv("./data/")

    # Load UtilityMatrix object from directory

    # um = UtilityMatrix.from_dir("./data/")

    # Access the SVT algorithm outside the UtilityMatrix class

    # fm, _ = SVT(um.ratings, max_iter=1500)

    # Initialize the recommender system and ask for recommendation with one of the three methods

    # r = QSRS(um)
    # print(r.recommendationV3(0, 5))

    # Test the 4 variants of the QSRS on the specified UtilityMatrix object
    # test_recco(r, um)

    # Test the matrix reconstruction of SVT on synthetic data

    # print(full_matrix_test(um))

    pass
