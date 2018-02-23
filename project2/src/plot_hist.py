#!/usr/bin/python

import argparse
import matplotlib.pyplot as plt
import numpy as np
import dataset

def plot_rating_hist(data, title, filename=None):
    """Plot histogram of ratings in the third column of data.

    data: an Nx3 numpy array of which only the last column will be used.
    title: A title to render on the plot.
    xlab: A label for the x-axis.
    filename: A file to write a PNG image to. If not provided, will plot to the device instead.

    """
    plt.close('all')
    ax = plt.figure().gca()
    plt.hist(data[:,2], bins=np.arange(0.5,6), rwidth=0.5)
    ax.set_title(title)
    ax.set_xlabel('Ratings')
    ax.set_ylabel('Frequency')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagebasename', type=str)
    args = parser.parse_args()

    ratings = dataset.load_ratings()

    def filename(suffix):
        return f'{args.imagebasename}_{suffix}.png' if args.imagebasename else None

    def ratings_of_movies(movie_ids):
        """Filter ratings down to those that rate the given movie IDs.

        Shuffles order from that in ratings to one sorted first by movie id, but this doesn't matter
        for producing histograms.

        """
        return np.concatenate([
            ratings[np.where(ratings[:,1] == id)]
            for id in movie_ids])

    # plot a histogram of all
    plot_rating_hist(ratings, 'All MovieLens Ratings', filename=filename('all'))

    # find the most popular movies
    top_ten_id = dataset.top_most_rated_movies(ratings, n=10)
    plot_rating_hist(
            ratings_of_movies(top_ten_id),
            'Ratings for top ten movies by rating frequency',
            filename=filename('top_ten_frequency'))

    # find the highest average rated movies
    top_ten_id = dataset.top_avg_rated_movies(ratings, n=10)
    plot_rating_hist(
            ratings_of_movies(top_ten_id),
            'Ratings for top ten movies by average rating',
            filename=filename('top_ten_rated'))

    # plot histoagram for all movies in three different genres
    for genre in ['Animation', 'Drama', 'Sci-Fi']:
        genre_movie_id = [m.id for m in dataset.Movie.query(genres=[genre])]
        plot_rating_hist(
            ratings_of_movies(genre_movie_id),
            f'Ratings of movies with genre {genre}',
            filename=filename(f'genre_{genre}'))
