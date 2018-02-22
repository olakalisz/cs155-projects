__doc__="""Latent Factor SVD as implemented with SGD for CS155 HW5."""

import numpy as np

def grad_U(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns one element of the sum in the gradient of the regularized loss function with
    respect to Ui, multiplied by eta.
    """
    return eta * (reg * Ui - Vj * (Yij - Ui.dot(Vj)))

def grad_V(Vj, Yij, Ui, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns one element of the sum in the gradient of the regularized loss function with
    respect to Vj, multiplied by eta.
    """
    return eta * (reg * Vj - Ui * (Yij - Ui.dot(Vj)))

def norm_fro(X):
    """Compute the Frobenius norm of a matrix X.

    Takes any numpy matrix X.

    Returns the scalar Frobenius norm.

    """
    return np.sqrt(sum([X[i,j] ** 2 for i in range(X.shape[0]) for j in range(X.shape[1])]))

def get_err(U, V, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    # regularization term
    err = 0.5 * reg * (norm_fro(U) ** 2 + norm_fro(V) ** 2)

    # square error
    for i, j, Yij in Y:
        err += 0.5 * (Yij - U[i].dot(V[j])) ** 2

    return err / len(Y)


def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300, verbose=False):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """
    U = np.random.uniform(-0.5, 0.5, (M, K))
    V = np.random.uniform(-0.5, 0.5, (N, K))
    err = np.zeros(max_epochs)
    for ep in range(max_epochs):
        # iterate through a random permutation of Y, adjusting relevant Ui and Vj for each
        for i, j, Yij in np.random.permutation(Y):
            dui = grad_U(U[i], Yij, V[j], reg, eta)
            dvj = grad_V(V[j], Yij, U[i], reg, eta)
            U[i] -= dui
            V[j] -= dvj
        err[ep] = get_err(U, V, Y, reg)
        if verbose:
            print(f'Epoch {ep} error: {err[ep]}')
        # early stopping condition
        if ep > 1 and err[ep - 1] - err[ep] < eps * (err[0] - err[0]):
            break

    return (U, V, get_err(U, V, Y))
