import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

def off_the_shelf_train(M, N, K, Y):
	"""
	Inputs:
	M - integer, number of users
	N - integer, number of movies
	K - integer, number of latent factors
	Y - a 2D array, first column is ith the user ID, second column is jth the movie ID, third column is the rating of user i on movie j

	Returns:
	(U, V)
	"""
	train_data_matrix = np.zeros((M,N))
	for n in range(Y_train.shape[0]):
    	i = Y_train[n,0] - 1
    	j = Y_train[n,1] - 1
    	yij = Y_train[n,2]
    	train_data_matrix[i][j] = yij

	U, s, V = svds(train_data_matrix, k = 20)
	s_diag_matrix=np.diag(s)
	X_pred = np.dot(np.dot(u, s_diag_matrix), V)

	return (U, V)