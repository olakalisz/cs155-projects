#!/usr/bin/python
import argparse
import matplotlib.pyplot as plt
import numpy as np

import dataset
import svd_sgd

def visualize_2d(M, index=None, labels=[], color=lambda id: 'black', alpha=1, filename=None):
    """Project a matrix into 2 dimensions and visualize it.

    If the input is mxn, produces a 2xn projection using the first two left singular vectors of M,
    and produces a scatterplot of the columns of this projection.

    If list index is provided, plots only the subset of columns indicated.

    If labels are provided, the indicated points are labeled in place on the graph (based on index
    matching between the labels list and the columns of the projection).

    color is a lambda that maps ids to a particular color for their point to be drawn in.

    alpha is a single value that is applied to all points drawn.

    If filename is provided, outputs the plot to the file indicated. Otherwise, outputs to the
    current matplotlib device.

    """
    # TODO consider mean-centering M
    A, sigma, B = np.linalg.svd(M)
    M_proj = np.matmul(A[:,:2].transpose(), M)
    # TODO: consider rescaling

    index = index or range(M.shape[1])
    plt.close('all')
    ax = plt.figure().gca()
    for i in index:
        ax.scatter(M_proj[0,i], M_proj[1,i], marker='.', c=color(i+1), alpha=alpha)
    for i, label in enumerate(labels):
        if i not in index:
            continue
        ax.annotate(label, M_proj[:,i])

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

    return M_proj

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagebasename', type=str)
    args = parser.parse_args()

    def filename(suffix):
        return f'{args.imagebasename}_{suffix}.png' if args.imagebasename else None

    # load and prepare data
    Y_train = dataset.load_ratings(source=dataset.RATINGS_TRAIN)
    Y_test = dataset.load_ratings(source=dataset.RATINGS_TEST)
    # get number of users, M, and number of movies, N, from distinct ids in the dataset
    M = len(set(Y_train[:,0]).union(set(Y_test[:,0])))
    N = len(set(Y_train[:,1]).union(set(Y_test[:,1])))
    # NB: we assume ids are consecutive integers up to M and N, and we change them to zero indexed
    Y_train[:,:2] -= np.ones((Y_train.shape[0], 2), dtype=int)
    Y_test[:,:2] -= np.ones((Y_test.shape[0], 2), dtype=int)
    movie_titles = [dataset.Movie(id).title for id in range(1, N+1)]

    # setting K=20 as specified in the assignment
    K = 20
    # TODO tune & justify choices of eta and reg
    eta = 0.03
    reg = 0.1

    # select movies that we will visualize in our projections: action and romance movies in top 30
    # by frequency of ratings
    movie_hist = np.histogram(Y_train[:,1], bins=range(1, M+2))[0]
    top_id = [id for _, id in sorted(zip(movie_hist, range(1, M+1)), reverse=True)[:30]]
    action_romance_id = [m.id for m in dataset.Movie.query(genres=['Romance', 'Action'])]
    # tranform to zero indexing to match transformed dataset
    action_romance_id = action_romance_id - np.ones(len(action_romance_id), dtype=int)
    action_romance_index = list(set(action_romance_id).intersection(set(top_id)))

    # visualize SVD as implemented for CS155 HW5
    U, V, _ = svd_sgd.train_model(M, N, K, eta, reg, Y_train, max_epochs=300)
    # visualize movies in 2d
    visualize_2d(
        V.transpose(), index=action_romance_index,
        labels=movie_titles, filename=filename('2d_movies_hw5'))
    # TODO visualize users too

    # TODO SVD with bias

    # TODO "off-the-shelf" SVD
