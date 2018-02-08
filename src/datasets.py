from base_models import common
import numpy as np
from sklearn.model_selection import KFold, train_test_split

# listed of trained models to stack - each references results stored in files
# named inferences/%s_{train,test}.txt, each of which may contain multiple
# columns for different hyperparameters
TRAINED_MODELS_FULL = [
    'adaboost',
    'adaboost_n374_lr1',
    'knn_2',
    'knn_4',
    'knn_8',
    'knn_16',
    'knn_32',
    'knn_64',
    'knn_128',
    'knn_256',
    'knn_512',
    'knn_1024',
    'rf',
    'rf2',
    'rf3',
    'ffm',
    'vw_logistic',
    'vw_logistic_nn5',
    'xgb',
    'xgb2',
    'lr',
    'lr2',
    'et',
    'nn',
    'nn2',
    'nn3',
]

# list of trained models with a 10% holdout
TRAINED_MODELS_HOLDOUT = [
    'adaboost_n290_lr1',
    'adaboost_n373_lr1',
    'knn_128_bc',
    'rf_md132_mss80',
    'rf_md144_mss80',
    'et',
]

# instantiating this here so that we can share the same random seed everywhere
KF_SEEDED = KFold(n_splits=4, shuffle=True, random_state=212)

def load_data(holdout=0.1):
    """Load provided training and test data from files.

    Returns numpy arrays X_train, y_train, X_test.

    """
    training_data = np.loadtxt('../data/training_data.txt', skiprows=1)
    X_test = np.loadtxt('../data/test_data.txt', skiprows=1)
    X_train = training_data[:,1:]
    y_train = training_data[:,0]
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_train, y_train, test_size=holdout, random_state=100)
    return X_train, y_train, X_test, X_holdout, y_holdout


def load_models(models=TRAINED_MODELS_HOLDOUT):
    """Load training results from specified models.

    Models should be strings that match values in TRAINED_MODELS_{FULL,HOLDOUT}.

    Returns aggregated numpy arrays containing a column for each model:
        X_meta_train, X_meta_test

    """
    # NB: knn models have multiple columns of results for different distance metrics, so we just
    # use the last column (bray curtis) which performed the best
    X_meta_train = np.hstack([
        np.loadtxt(f'../inferences/{m}_train.txt', ndmin=2)[:,-1:] for m in models])
    X_meta_test = np.hstack([
        np.loadtxt(f'../inferences/{m}_test.txt', ndmin=2)[:,-1:] for m in models])
    return X_meta_train, X_meta_test


def model_accuracy(models=TRAINED_MODELS_HOLDOUT, holdout=0.1):
    """Compute the model accuracy for each of the specified models.

    The holdout parameter indicates what percent holdout the indicated models were trained with,
    and effects what subset of y_train the data is compared with.

    Returns a list of same size as the models parameter, with each element being a float accuracy.

    """
    # NB: this function currently works with models from before we trained with a holdout. it will
    # need to be changed to work with the 18000-sized datasets
    _, y_train, _, _, _ = load_data(holdout=holdout)
    X_meta_train, _ = load_models(models=models)
    assert X_meta_train.shape[1] == len(models), X_meta_train.shape[1]
    accuracies = []
    for i in range(len(models)):
        accuracies.append(common.train_accuracy(X_meta_train[:,i], y_train))

    return accuracies
