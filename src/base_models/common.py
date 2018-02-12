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
        title=None, param_lab='', output=None):
    """Plot the learning curve for a given model vs. one parameter.

    model should be a callable with signature:
      (X_train, y_train, X_test, param),
    where param is the parameter we're varying here, and returns two numpy arrays:
      train_class, test_class.

    kf is a keras kfold class to define how we split for cross validation.

    param_range is a list of values to train the model for and plot.

    title and param_lab are used as labels in the graph that is produced.

    output is an IO object which, if specified, will be written to with progress data

    Plots the graph to the a PNG file named as the title, and returns a numpy array of training
    and validation accuracy vs. param.

    """
    title = title or 'Learning Rate of {}' % model.__name__
    kf = kf or datasets.KF_SEEDED
    acc_vs_p = np.zeros((len(param_range), 3))
    for i, p in enumerate(param_range):
        if output:
            output.write(f'Training for p={p}\n')
        train_acc = val_acc = 0.0
        for j, (train_index, test_index) in enumerate(kf.split(range(X_train.shape[0]))):
            insample_class, _, test_class = model(
                X_train[train_index], y_train[train_index], X_train[test_index], p)
            # keep an average of accuracies seen across all folds
            train_acc = (train_acc * j + train_accuracy(insample_class, y_train[train_index])) / (j+1)
            val_acc = (val_acc * j + train_accuracy(test_class, y_train[test_index])) / (j+1)
        acc_vs_p[i] = np.array([p, train_acc, val_acc])
        if output:
            output.write(f'Train acc={train_acc:.3}\n')
            output.write(f'Valid acc={val_acc:.3}\n')
            # flush output in case we have a crash
            output.flush()

    # plot the results
    plt.plot(acc_vs_p[:,0], acc_vs_p[:,1], label='Training Accuracy')
    plt.plot(acc_vs_p[:,0], acc_vs_p[:,2], label='Validation Accuracy')
    plt.title(title)
    plt.xlabel(param_lab)
    plt.ylabel('Average Accuracy')
    plt.legend()
    plt.savefig(title.replace(' ','_') + '.png')
    plt.close()

    return acc_vs_p
