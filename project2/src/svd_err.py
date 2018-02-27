#!/usr/bin/python
import argparse
import heapq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD, SVDpp

import dataset
import off_the_shelf
import svd_sgd

if __name__ == '__main__':
    # load and prepare data
    # NB: ratings_all stays 1-indexed, while Y_train and Y_test are transformed to be 0-indexed
    ratings_all = dataset.load_ratings(source=dataset.RATINGS_FULL)
    Y_train = dataset.load_ratings(source=dataset.RATINGS_TRAIN)
    Y_test = dataset.load_ratings(source=dataset.RATINGS_TEST)
    # get number of users, M, and number of movies, N, from distinct ids in the dataset
    M = len(set(Y_train[:,0]).union(set(Y_test[:,0])))
    N = len(set(Y_train[:,1]).union(set(Y_test[:,1])))
    # NB: we assume ids are consecutive integers up to M and N, and we change them to zero indexed
    Y_train[:,:2] -= np.ones((Y_train.shape[0], 2), dtype=int)
    Y_test[:,:2] -= np.ones((Y_test.shape[0], 2), dtype=int)
    sparse_matrix = dataset.construct_user_movie_matrix(source=dataset.RATINGS_TRAIN, M=M, N=N)

    # setting K=20 as specified in the assignment
    K = 20
    eta = 0.03
    reg = 1

    # visualize SVD as implemented for CS155 HW5
    U, V, e_in = svd_sgd.train_model(M, N, K, eta, reg, Y_train, max_epochs=300)
    e_out = svd_sgd.get_err(U, V, Y_test)
    print(f'HW5 Implementation In-Sample Error: {e_in:.3}')
    print(f'HW5 Implementation Out-of-Sample Error: {e_out:.3}')

    # "off-the-shelf" SVD from numpy
    U, Sigma, V = off_the_shelf.scipy_svd_train(M, N, K, Y_train)
    U = np.matmul(U, np.diag(np.sqrt(Sigma)))
    V = np.matmul(np.diag(np.sqrt(Sigma)), V)
    e_in = svd_sgd.get_err(U, V.transpose(), Y_train)
    e_out = svd_sgd.get_err(U, V.transpose(), Y_test)
    print(f'SciPy SVD In-Sample Error: {e_in:.3}')
    print(f'SciPy SVD Out-of-Sample Error: {e_out:.3}')

    # Surprise models
    svd_models = [
        ('SVD Unbiased', SVD(n_factors=20, biased=False, n_epochs=100)),
        ('SVD w/ Global and Term Bias', SVD(n_factors=20, n_epochs=100)),
        ('SVD++', SVDpp(n_factors=20)),
    ]
    def get_surprise_err(model, d):
        err = 0.0
        for u, v, rating in d:
            # square error
            est = model.predict(u, v).est
            err += 0.5 * (int(rating) - est) ** 2

        return err / len(d)
    reader = Reader(line_format='user item rating', sep='\t')
    data = Dataset.load_from_folds([('../data/train.txt','../data/test.txt')], reader)
    train, test = list(data.folds())[0]

    for label, svd in svd_models:
        svd.fit(train)
        e_in = get_surprise_err(svd, Y_train)
        e_out = get_surprise_err(svd, test)
        print(f'Surprise {label} In-Sample Error: {e_in:.3}')
        print(f'Surprise {label} Out-of-Sample Error: {e_out:.3}')
