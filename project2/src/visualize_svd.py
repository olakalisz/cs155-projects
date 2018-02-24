#!/usr/bin/python
import argparse
import heapq
import matplotlib.pyplot as plt
import numpy as np

import dataset
import off_the_shelf
import svd_sgd

def visualize_2d(M, index=None, labels=[], color=lambda id: 'black', alpha=1,
        label_outliers=None, title='2D Projection', legend=None, filename=None):
    """Project a matrix into 2 dimensions and visualize it.

    If the input is mxn, produces a 2xn projection using the first two left singular vectors of M,
    and produces a scatterplot of the columns of this projection.

    If list index is provided, plots only the subset of columns indicated.

    If labels are provided, the indicated points are labeled in place on the graph (based on index
    matching between the labels list and the columns of the projection).

    color is a lambda that maps ids to a particular color for their point to be drawn in.

    alpha is a single value that is applied to all points drawn.

    label_outliers, if provided, will specify a number of points to find that are of max distance
    from the origin in the 2d projection, and only label those points.

    title is passed to matplotlib for use in rendering the figure title.

    legend, if provided, is passed to the matplotlib legend command.

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

    # find outliers
    outliers = []
    if label_outliers:
        for i in range(M.shape[1]):
            # tuples beginning with negative distance in a minheap will result in a maxheap on
            # actual distance
            heapq.heappush(outliers, (-1 * (M_proj[0,i]**2 + M_proj[1,i]**2), i))
        top_outliers = [i for _, i in [heapq.heappop(outliers) for _ in range(label_outliers)]]

    for i, label in enumerate(labels):
        if i not in index:
            continue
        if label_outliers and i not in top_outliers:
            continue

        ax.annotate(label, M_proj[:,i])

    ax.set_title(title)
    if legend:
        ax.legend(legend[0])
        for handle, c in zip(ax.get_legend().legendHandles, legend[1]):
            handle.set_color(c)

    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def produce_plots(V, method_name, ratings_all, M, N, imagebasename=None, singlegenreplot=False):
    """Produce the six plots specified in the homework for the given V matrix.

    Specifically, produces plots of the following to either files with the given prefix or device:
        * Action and Romance movies occurring in the top thirty most rated movies
        * Top ten movies by number of ratings
        * Top ten movies by average rating
        * Ten movies each from genres Animation, Drama, Sci-Fi

    """
    def filename(suffix):
        return f'{imagebasename}_{method_name}_{suffix}.png' if imagebasename else None

    movie_titles = [dataset.Movie(id).title for id in range(1, N+1)]

    # project action and romance movies in top 30 by frequency of ratings
    top_thirty_id = list(dataset.top_most_rated_movies(ratings_all, n=30) - np.ones(30, dtype=int))
    action_romance_id = [m.id for m in dataset.Movie.query(genres=['Romance', 'Action'])]
    # tranform to zero indexing to match transformed dataset
    action_romance_id = action_romance_id - np.ones(len(action_romance_id), dtype=int)
    action_romance_index = list(set(action_romance_id).intersection(set(top_thirty_id)))
    visualize_2d(
        V, index=action_romance_index,
        labels=movie_titles, title=f'{method_name} Projection of Action/Romance Movies',
        filename=filename('action_romance'))

    # labeling ten outliers
    visualize_2d(
        V, label_outliers=10, labels=movie_titles,
        title=f'{method_name} with ten outliers labeled',
        filename=filename('outliers'))

    # find the most popular movies
    top_ten_id = list(dataset.top_most_rated_movies(ratings_all, n=10) - np.ones(10, dtype=int))
    visualize_2d(
        V, index=top_ten_id, labels=movie_titles,
        title=f'{method_name} with ten most rated movies labeled',
        filename=filename('top_ten_frequent'))

    # find the highest average rated movies
    top_ten_id = list(dataset.top_avg_rated_movies(ratings_all, n=10) - np.ones(10, dtype=int))
    visualize_2d(
        V, index=top_ten_id, labels=movie_titles,
        title=f'{method_name} with ten highest rated movies labeled',
        filename=filename('top_ten_rated'))

    # label top ten movies in three different genres
    if singlegenreplot:
        genre_movie_id = []
        movie_colors = {} # map from movie id to color based on genre
        colors = ['blue', 'yellow', 'red']
        for genre, color in zip(['Animation', 'Drama', 'Sci-Fi'], colors):
            new_ids = [m.id for m in dataset.Movie.query(genres=[genre])[:10]]
            genre_movie_id += new_ids
            for id in new_ids:
                movie_colors[id] = color
        genre_movie_id = list(genre_movie_id - np.ones(len(genre_movie_id), dtype=int))
        visualize_2d(
            V, index=genre_movie_id, labels=movie_titles,
            title=f'{method_name} with ten movies from three different genres',
            color=lambda id: movie_colors[id],
            legend=[('Animation', 'Drama', 'Sci-Fi'), colors],
            filename=filename('all_genres'))
    else:
        for genre in ['Animation', 'Drama', 'Sci-Fi']:
            genre_movie_id = [m.id - 1 for m in dataset.Movie.query(genres=[genre])[:10]]
            visualize_2d(
                V, index=genre_movie_id, labels=movie_titles,
                title=f'{method_name} with ten movies from genre {genre}',
                filename=filename(genre))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagebasename', type=str)
    parser.add_argument('--singlegenreplot', action='store_true', default=False)
    args = parser.parse_args()

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
    U, V, _ = svd_sgd.train_model(M, N, K, eta, reg, Y_train, max_epochs=300)
    produce_plots(V.transpose(), 'HW5', ratings_all, M, N, args.imagebasename, args.singlegenreplot)

    # "off-the-shelf" SVD from numpy
    U, V = off_the_shelf.scipy_svd_train(M, N, K, Y_train)
    produce_plots(V, 'SciPy', ratings_all, M, N, args.imagebasename, args.singlegenreplot)
