# cs155-projects
Caltech CS 155 Winter 2018

Code for projects in [Caltech's CS155: Machine Learning and Data Mining](http://www.yisongyue.com/courses/cs155/2018_winter/), developed by team [Aw Young Qingzhuo](https://github.com/veniversum), [Ola Kalisz](https://github.com/olakalisz), and [Riley Patterson](https://github.com/rylz).

### Miniproject 1: Amazon Review Sentiment Detection

The first project is a Kaggle competition to detect sentiment in Amazon reviews. Details on the task are provided on the [Kaggle project page](https://www.kaggle.com/c/caltech-cs-155-2018), and our code is in the [src directory](project1/src), in particular in [a jupyter notebook](project1/src/generate_first_layer.ipynb) for training individual models and in [this source file defining a stacked neural net using these results](project1/src/stacked_neural_net_model.py).

Initial summaries of results for individual models were tracked in [this issue](../../issues/1), and further results will be in project report.

### Miniproject 2: Visualizing Matrix Factorizations for Movie Ratings

The second project involves finding matrix factorizations for the MovieLens dataset and projecting it into two dimensions for visualization and interpretation of how and why movies differ from each other.

We applied three different implementations of matrix factorization:
* An implementation of SVD in which we decompose into two matrices, U and V, which incorporate the singular values into the matrices themselves.
* SVD with Bias terms.
* SVD from SciPy from which we analyze matrices U and V which _do not_ incorporate the singular values.

For each of these, we produced 2D visualizations with various colorings and labelings to help infer what the two extracted dimensions correlated to.

The code for this project is in [the source directory for project2](project2/src), and the results are found in [the project2 report directory](project2/report).
