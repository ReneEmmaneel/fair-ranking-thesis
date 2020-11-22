#!/usr/bin/env python

#In this file, sklearn is used to create rankSVM
#The implementation is based on the implementation by Fabian Pedregosa and Alexandre Gramfort
#See https://gist.github.com/agramfort/2071994

import itertools
import numpy as np
import argparse, os
import matplotlib.pyplot as plt
from sklearn import svm, linear_model, model_selection, datasets, preprocessing
from warnings import filterwarnings
import pandas as pd
import pickle
filterwarnings('ignore')

def import_libsvm(file):
    """Given a file in libsvm format, transform it into pairs of relevant/irrelevant
    In this method, all pairs are choosen, except for those that have the
    same relevance value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1
    Parameters
    ----------
    file : file in libsvm format. Libsvm format is n lines with relevance and feature vector
    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    data = datasets.load_svmlight_file(file)
    X = data[0].toarray()
    y = data[1]

    X_new = []
    y_new = []
    comb = itertools.combinations(range(len(X)), 2)
    for k, (i, j) in enumerate(comb):
        if y[i] == y[j] or (not X[i][0] == X[j][0]):
            continue
            # skip if same relevance or not the same query
        X_new.append(X[i][1:] - X[j][1:])
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

    def fit(self, X_trans, y_trans, max_iter = 1000):
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
        self.max_iter = max_iter
        super(RankSVM, self).fit(X_trans, y_trans)
        return self

    def fit_and_plot(self, X_trans_train, y_trans_train, X_trans_test, y_trans_test):
        for i in range(100):
            self.max_iter = 20

        # Create a template lit to store accuracies
        acc = []

        iterations = np.logspace(0,5, num = 6)

        # Iterate along a logarithmically spaced ranged
        for i in iterations:
            # Print out the number of iterations to use for the current loop
            # Create an SVC algorithm with the number of iterations for the current loop
            self.max_iter = i
            # Fit the algorithm to the data
            super(RankSVM, self).fit(X_trans_train, y_trans_train)
            # Append the current accuracy score to the template list
            score = self.score(X_trans_test, y_trans_test)
            print('Score after {} iterations: {}'.format(int(i),score))
            acc.append(score)

        # Convert the accuracy list to a series
        acc = pd.Series(acc, index = iterations)
        # Set the plot size
        plt.figure(figsize = (15,10))
        # Set the plot title
        title = 'Graph to show the accuracy of the SVC model as number of iterations increases\nfinal accuracy: ' + str(acc.iloc[-1])
        plt.title(title)
        # Set the xlabel and ylabel
        plt.xlabel('Number of iterations')
        plt.ylabel('Accuracy score')
        # Plot the graph
        acc.plot.line()
        plt.show()



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
    parser.add_argument("-f", "--file", dest="libsvm_file", required=False,
                        help="training data in libsvm format", metavar="FILE")
    parser.add_argument("-p", "--pickle", dest="pickle_folder", required=False,
                        help="dump pickle info", metavar="FOLDER")
    parser.add_argument("--score", dest="score", required=False, action="store_true",
                        help="score pickled model")

    args = parser.parse_args()

    if args.pickle_folder and args.score:
        model = pickle.load(open(args.pickle_folder + 'model', 'rb'))
        X_trans = pickle.load(open(args.pickle_folder + 'X_trans', 'rb'))
        y_trans = pickle.load(open(args.pickle_folder + 'y_trans', 'rb'))
        print('Performance of ranking {}'.format(model.score(X_trans, y_trans)))
    else:
        if args.libsvm_file:
            X_trans, y_trans = import_libsvm(args.libsvm_file)
            #X_trans = preprocessing.scale(X_trans)

            if args.pickle_folder:
                pickle.dump(X_trans, open(args.pickle_folder + 'X_trans', 'wb'))
                pickle.dump(y_trans, open(args.pickle_folder + 'y_trans', 'wb'))
        elif args.pickle_folder:
            X_trans = pickle.load(open(args.pickle_folder + 'X_trans', 'rb'))
            y_trans = pickle.load(open(args.pickle_folder + 'y_trans', 'rb'))
        else:
            print('Specify either libsvm_file or picle_folder')
            os._exit(0)

        print('loaded! now for training')
        X_trans_train, X_trans_test, y_trans_train, y_trans_test = model_selection.train_test_split(X_trans, y_trans, train_size=0.5)

        #Train the model, and print the performance of the model. If you want to plot the performance over iterations, use:
        #RankSVM().fit_and_plot(X_trans_train, y_trans_train, X_trans_test, y_trans_test)
        rank_svm = RankSVM().fit(X_trans_train, y_trans_train, max_iter = 10000)
        print('Performance of ranking {}'.format(rank_svm.score(X_trans_test, y_trans_test)))

        if args.pickle_folder:
            model = pickle.dump(rank_svm, open(args.pickle_folder + 'model', 'wb'))
