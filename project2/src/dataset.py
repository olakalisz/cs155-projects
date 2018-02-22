import numpy as np

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
