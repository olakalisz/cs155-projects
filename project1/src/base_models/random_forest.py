import datasets
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def RandomForest(X_train, y_train, X_test, n_estimators=250, max_depth=144, min_samples_split=80,
        kf=datasets.KF_SEEDED, verbose=False):
    """Train a single random forest model on the data with k-fold cross validation.

    Returns three numpy arrays:
      * majority in-sample classification from the k-folds
      * training classifications from cross-validation
      * test classifications

    """
    rf_classes_insample = np.zeros(y_train.shape)
    rf_classes_train = np.empty(y_train.shape)
    rf_classes_test = np.empty(X_test.shape[0])

    for split, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
        if verbose:
            print('RandomForest, split=%d' % (split))
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_features='sqrt', criterion='gini',
            max_depth=max_depth, min_samples_split=min_samples_split)
        clf.fit(X_train[train_index], y_train[train_index])
        rf_classes_train[test_index] = clf.predict(X_train[test_index])
        rf_classes_insample[train_index] += clf.predict(X_train[train_index]) / kf.n_splits

    if verbose:
        print('RandomForest, test')
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_features='sqrt', criterion='gini',
        max_depth=max_depth, min_samples_split=min_samples_split)
    clf.fit(X_train, y_train)
    rf_classes_test = clf.predict(X_test)

    return np.rint(rf_classes_insample), rf_classes_train, rf_classes_test
