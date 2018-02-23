import numpy as np
import re

# movie metadata encoded as indicated in MOVIE_FEATURES
MOVIES = '../data/movies.txt'
# user ratings encoded as user_id, movie_id, rating
RATINGS_FULL = '../data/data.txt'
RATINGS_TRAIN = '../data/train.txt'
RATINGS_TEST = '../data/test.txt'

MOVIE_FEATURES = [
    'id',
    'title',
    '', # unknown feature
    'Action',
    'Adventure',
    'Animation',
    'Childrens',
    'Comedy',
    'Crime',
    'Documentary',
    'Drama',
    'Fantasy',
    'Film-Noir',
    'Horror',
    'Musical',
    'Mystery',
    'Romance',
    'Sci-Fi',
    'Thriller',
    'War',
    'Western',
]

YEAR_RE = re.compile('\((\d+)\)')
class Movie:
    # class property to store dataset once
    _movies = {}

    def __init__(self, id):
        self.id = id
        raw_data = self.movies()[id]
        self.title = raw_data[0]
        self.genres = []
        for genre, indicator in zip(MOVIE_FEATURES[3:], raw_data[2:]):
            if int(indicator):
                self.genres.append(genre)

    @property
    def year(self):
        match = YEAR_RE.search(self.title)
        if match:
            return int(match.group(1))

        # dumb default
        print(self.title)
        return 1950

    @classmethod
    def query(cls, genres=[]):
        """Query all movies that match certain criteria.

        If genre is provided, returns objects of movies that match any genre in that list.

        """
        return [Movie(id) for id in cls.movies() if set(genres).intersection(set(Movie(id).genres))]

    @classmethod
    def movies(cls):
        """A dictionary of all movie data.

        The dictionary is keyed on integer movie ids and contains string values mapping to
        MOVIE_FEATURES defined above.

        """
        if not cls._movies:
            cls._movies = {
                int(r[0]): r[1:]
                for r in np.loadtxt(MOVIES, dtype=str, delimiter='\t', encoding='latin1')
            }
        return cls._movies

def load_ratings(source=RATINGS_FULL):
    """Load ratings from a specified source file.

    Assumes the MovieLens rating format, rows of:
        (user id, movie id, rating)

    Returns a Nx3 numpy array containing that data.

    """
    return np.loadtxt(source, dtype=int, delimiter='\t')

def construct_user_movie_matrix(source=RATINGS_FULL, M=None, N=None):
    """Load ratings data and construct a sparse #usersX#movies matrix of ratings.

    M and N are #users and #movies, respectively. If not provided, they will be inferred from data.

    User and movie ids are assumed to be sequential integers which start at 1.

    """
    ratings = load_ratings(source)
    M = M or len(set(ratings[:,0]))
    N = N or len(set(ratings[:,1]))
    mat = np.zeros((M, N), dtype=int)
    for user, movie, rating in ratings:
        mat[user-1,movie-1] = rating
    return mat

def top_most_rated_movies(ratings, n=10, M=None):
    """Returns movie ids for the top n movies by number of times they have been rated.

    M is the number of movies. If not provided, it will be inferred from the dataset based on an
    assumption that ids are sequential starting from 1.

    """
    M = M or len(set(ratings[:,1]))
    # first find a histogram with M buckets for every integer
    movie_hist = np.histogram(ratings[:,1], bins=range(1, M+2))[0]
    return [id for _, id in sorted(zip(movie_hist, range(1, M+1)), reverse=True)[:n]]

def top_avg_rated_movies(ratings, n=10, M=None):
    """Returns movie ids for the top n movies by average rating.

    M is the number of movies. If not provided, it will be inferred from the dataset based on an
    assumption that ids are sequential starting from 1.

    """
    M = M or len(set(ratings[:,1]))
    # note that this takes M**2 time unnecessarily but M=1682 so not worrying about it
    avg_ratings = [np.mean(ratings[np.where(ratings[:,1] == i),2]) for i in range(1, M+1)]
    return [id for _, id in sorted(zip(avg_ratings, range(1, M+1)), reverse=True)[:n]]

# hand bucketing genres into three buckets so that we can map to rgb
GENRE_BUCKETS = {
    'artsy': [
        'Documentary',
        'Drama',
        'Film-Noir',
        'Musical',
    ],
    'light': [
        'Adventure',
        'Animation',
        'Childrens',
        'Comedy',
        'Fantasy',
        'Romance',
    ],
    'actionlike': [
        'Action',
        'Crime',
        'Horror',
        'Mystery',
        'Sci-Fi',
        'Thriller',
        'War',
        'Western',
    ],
}

def genres_to_rgb(movie_id):
    """Map movie to RGB based on number of genres that fall in the buckets defined above.

    Returns a 3-tuple of floats in range [0.0, 1.0] indicating amount of red, green, blue.

    """
    return tuple([
        len(set(bucket).intersection(set(Movie(movie_id).genres))) / len(bucket)
        for bucket in GENRE_BUCKETS.values()])
