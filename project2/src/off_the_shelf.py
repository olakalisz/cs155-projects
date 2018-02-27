import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

def scipy_svd_train(M, N, K, Y):
    """Return a decomposition of Y using SciPy SVD implementation.

    Inputs:
    M - integer, number of users
    N - integer, number of movies
    K - integer, number of latent factors
    Y - a 2D array, first column is ith the user ID, second column is jth the movie ID, third column is the rating of user i on movie j

    Returns:
    (U, Sigma, V)
    """
    train_data_matrix = np.zeros((M,N))
    for n in range(Y.shape[0]):
        i = Y[n,0]
        j = Y[n,1]
        yij = Y[n,2]
        train_data_matrix[i][j] = yij

    U, s, V = svds(train_data_matrix, k = 20)

    return (U, s, V)
