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
	for n in range(Y.shape[0]):
    	i = Y[n,0] - 1
    	j = Y[n,1] - 1
    	yij = Y[n,2]
    	train_data_matrix[i][j] = yij

	U, s, V = svds(train_data_matrix, k = 20)
	s_diag_matrix=np.diag(s)
	X_pred = np.dot(np.dot(U, s_diag_matrix), V)

	return (U, V)