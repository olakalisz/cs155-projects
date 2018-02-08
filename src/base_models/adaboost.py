from . import common
import datasets
import numpy as np
from sklearn.ensemble import AdaBoostClassifier

def AdaBoost(X_train, y_train, X_test, n_estimators=250, learning_rate=1, kf=datasets.KF_SEEDED,
        verbose=False):
    """Train a single adaboost model on the data.

    Returns two numpy arrays: training and test predictions.

    """
    adaboost_classes_train = np.empty(y_train.shape)
    adaboost_classes_test = np.empty(X_test.shape[0])

    for split, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
        if verbose:
            print('Adaboost, split=%d' % (split))
        clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        clf.fit(X_train[train_index], y_train[train_index])
        adaboost_classes_train[test_index] = clf.predict_proba(X_train[test_index])[:, 1]

    if verbose:
        print('Adaboost, test')
    clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    clf.fit(X_train, y_train)
    adaboost_classes_test = clf.predict_proba(X_test)[:, 1]

    return adaboost_classes_train, adaboost_classes_test
