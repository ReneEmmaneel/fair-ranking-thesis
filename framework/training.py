#!/usr/bin/env python

#In this file, sklearn is used to create rankSVM
#The implementation is based on the implementation by Fabian Pedregosa and Alexandre Gramfort
#See https://gist.github.com/agramfort/2071994

import itertools
import numpy as np
import argparse
from sklearn import svm, linear_model, model_selection, datasets, preprocessing

def transform_pairwise(X, y):
    """Transforms data into pairs with balanced labels for ranking
    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.
    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.
    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    X_new = []
    y_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]

    return np.asarray(X_new), np.asarray(y_new).ravel()

def import_libsvm(file):
    data = datasets.load_svmlight_file(file)
    X = data[0].toarray()
    y = data[1]

    X_new = []
    y_new = []
    comb = itertools.combinations(range(len(X)), 2)
    for k, (i, j) in enumerate(comb):
        if y[i] == y[j]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i] - y[j]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]

    return np.asarray(X_new), np.asarray(y_new).ravel()

class RankSVM(svm.LinearSVC):
    """Performs pairwise ranking with an underlying LinearSVC model
    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.
    See object :ref:`svm.LinearSVC` for a full description of parameters.
    """

    def fit(self, X_trans, y_trans):
        """
        Fit a pairwise ranking model.
        Parameters
        ----------
        X_trans : array, shape (k, n_feaures)
            Data as pairs
        y_trans : array, shape (k,)
            Output class labels, where classes have values {-1, +1}
        -------
        self
        """
        self.max_iter = 1000
        self.verbose = 1
        super(RankSVM, self).fit(X_trans, y_trans)
        return self

    def predict(self, X):
        """
        Predict an ordering on X. For a list of n samples, this method
        returns a list from 0 to n-1 with the relative order of the rows of X.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        Returns
        -------
        ord : array, shape (n_samples,)
            Returns a list of integers representing the relative order of
            the rows in X.
        """
        if hasattr(self, 'coef_'):
            np.argsort(np.dot(X, self.coef_.T))
        else:
            raise ValueError("Must call fit() prior to predict()")

    def score(self, X_trans, y_trans):
        """
        Because we transformed into a pairwise problem, chance level is at 0.5
        """
        return np.mean(super(RankSVM, self).predict(X_trans) == y_trans)


if __name__ == '__main__':
    # as showcase, we will create some non-linear data
    # and print the performance of ranking vs linear regression

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", dest="libsvm_file", required=True,
                        help="training data in libsvm format", metavar="FILE")

    args = parser.parse_args()

    X_trans, y_trans = import_libsvm(args.libsvm_file)
    X_trans = preprocessing.scale(X_trans)
    print(X_trans[0:25])

    print('loaded! now for training')
    kf = model_selection.KFold(2)
    # print the performance of ranking
    for train_index, test_index in kf.split(X_trans):
        rank_svm = RankSVM().fit(X_trans[train_index], y_trans[train_index])
        print('Performance of ranking {}'.format(rank_svm.score(X_trans[test_index], y_trans[test_index])))
