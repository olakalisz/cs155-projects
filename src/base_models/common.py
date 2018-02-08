import datasets
import matplotlib.pyplot as plt
import numpy as np

def train_accuracy(predictions, y_train):
    """Compute prediction accuracy for binary classification for a set of trained results.

    Both inputs should be same-sized numpy column vectors.

    Returns a single float representing accuracy across all samples.

    """
    return (1 - np.sum(np.abs((np.rint(predictions)) - y_train)) / y_train.shape[0])

def plot_learning_curve(model, X_train, y_train, kf=None, param_range=[],
        title=None, param_lab=''):
    """Plot the learning curve for a given model vs. one parameter.

    model should be a callable with signature:
      (X_train, y_train, X_test, param),
    where param is the parameter we're varying here, and returns two numpy arrays:
      train_class, test_class.

    kf is a keras kfold class to define how we split for cross validation.

    param_range is a list of values to train the model for and plot.

    title and param_lab are used as labels in the graph that is produced.

    """
    title = title or 'Learning Rate of {}' % model.__name__
    kf = kf or datasets.KF_SEEDED
    error_vs_p = np.zeros((len(param_range), 3))
    for i, p in enumerate(param_range):
        train_err = val_err = 0.0
        for j, (train_index, test_index) in enumerate(kf.split(range(X_train.shape[0]))):
            train_class, test_class = model(
                X_train[train_index], y_train[train_index], X_train[test_index], p)
            # keep an average of errors seen across all folds
            train_err = (train_err * j + train_accuracy(train_class, y_train[train_index])) / (j+1)
            val_err += (val_err * j + train_accuracy(test_class, y_train[test_index])) / (j+1)
        error_vs_p[i] = np.array([N, train_err, val_err])

    # plot the results
    #plt.axis([20, 100, 0, 6])
    plt.plot(error_vs_N[:,0], error_vs_N[:,1], label='Training Error')
    plt.plot(error_vs_N[:,0], error_vs_N[:,2], label='Validation Error')
    plt.set_title(title)
    # NB: this only produces labels on the last plot; why it doesn't work on all is unknown
    plt.xlabel(param_lab)
    plt.ylabel('Average Accuracy')
    plt.legend()
    plt.show()
