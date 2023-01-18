import pandas as pd
import numpy as np
from numpy.linalg import norm as np_norm
import scipy.sparse as ss
from scipy.sparse.linalg import norm as ss_norm
from scipy.sparse.linalg import svds
from sparsesvd import sparsesvd
import matplotlib.pyplot as plt
import math, random
from tqdm import tqdm


def SVT(M: pd.DataFrame,
        max_iter: int = 1500,
        delta: int = 2,
        tolerance: float = 0.001,
        increment: int = 5):
    """
    Params:
        M: matrix to complite
        max_iter: maximum number of iterations
        delta: step-size
        tolerance: tolerance on the minimum improvement
        increment: how many new singular values to check if they fall below tau
    Returns:
        X, rmse: complited matrix, error list
    """
    M = ss.csr_matrix(M.fillna(0).values)  # pandas DF into scipy sparse matrix
    n, m = M.shape

    total_num_nonzero = len(M.nonzero()[0])
    idx = random.sample(range(total_num_nonzero), int(total_num_nonzero))
    Omega = (M.nonzero()[0][idx], M.nonzero()[1][idx])

    tau = 5 * math.sqrt(n * m)

    ######
    # SVT
    ######
    r = 0
    rmse = []
    data, indices = np.ravel(M[Omega]), Omega
    P_Omega_M = ss.csr_matrix((data, indices), shape=(n, m))
    k_0 = np.ceil(tau / (delta * ss_norm(P_Omega_M)))  # element-wise ceiling
    Y = k_0 * delta * P_Omega_M

    for _ in tqdm(range(max_iter), desc= "Iteratively filling the matrix", colour="green"):
        s = r + 1
        while True:
            U, S, V = sparsesvd(ss.csc_matrix(Y), s)
            s += increment
            try:
                if S[s - increment] <= tau: break
            except:
                break

        r = np.sum(S > tau)

        U = U.T[:, :r]
        S = S[:r] - tau
        V = V[:r, :]
        X = (U * S).dot(V)
        X_omega = ss.csr_matrix((X[Omega], Omega), shape=(n, m))

        if ss_norm(X_omega - P_Omega_M) / ss_norm(P_Omega_M) < tolerance: break

        diff = P_Omega_M - X_omega
        Y += delta * diff
        rmse.append(np_norm(M[M.nonzero()] - X[M.nonzero()]) / np.sqrt(len(X[M.nonzero()])))
        X.clip(0, 1)

    return X, rmse


if __name__ == "__main__":
    ############
    # Read data
    ############

    utility_table_path = "./data/utility_matrix.csv"

    utility_df = pd.read_csv(utility_table_path, index_col=0)
    utility_df = utility_df
    print(utility_df.head(15))

    ##################################
    # Compute SVT and visualize error
    ##################################

    X, rmse = SVT(utility_df, max_iter=1500)
    print(pd.DataFrame(X).head(15))

    x_coordinate = range(len(rmse))
    plt.ylim(0, 0.5)
    plt.xlabel('Number of iterations')
    plt.ylabel('RMSE')
    plt.plot(x_coordinate, rmse, '-')
    plt.show()
