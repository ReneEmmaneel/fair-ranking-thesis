#!/usr/bin/env python

# File containing various ranking functions for ranking models.
# It contains both functions for ranking and for fair ranking.

# The functions (to be) implemented:
# -Discounted Cumulative Gain
# -Mean reciprocal rank
# -Fair ranking functions?

import numpy as np
import argparse, os
import pandas as pd
import training
from training import RankSVM
from sklearn import datasets
import pickle
import math

def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

def import_libsvm_and_rank(model, in_file):
    """Given a file in libsvm format, rerank the file into a ranked list of
    estimated relevance values, and write it to the given file

    Parameters
    ----------
    model : rankSVM model (see training.py)
    in_file : file in libsvm format. Libsvm format is n lines with relevance and feature vector

    Output
    ----------
    ranked_list:
        [[qid1, [rel1, rel2...], [feature_vec1, feature_vec2...],
         ...
        ]

    """
    data = datasets.load_svmlight_file(in_file, query_id = True)
    X = data[0].toarray()
    y = data[1]
    qid = data[2]

    # step 1, make a ranked_list with qid, rel_vector, feature_vec
    ranked_list = []
    for i, x in enumerate(X):
        #if qid already exists, add to the list
        added = False
        for query in ranked_list:
            if query[0] == qid[i]:
                query[1].append((y[i], x))
                added = True
                break
        #else add the qid
        if not added:
            ranked_list.append([qid[i], [(y[i], x)]])

    #step 2, reranking; for every query rank the feature vectors using the given model
    for i, query in enumerate(ranked_list):
        query[1] = sorted(query[1], key=cmp_to_key(sort_documents, model))

    return ranked_list

def sort_documents(a,b, model):
	return model.predict_single(b[1], a[1])

def cmp_to_key(cmp_fun, model):
    """Convert a cmp= function into a key= function
    see: https://docs.python.org/3.6/howto/sorting.html#the-old-way-using-the-cmp-parameter
    """
    class K:
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return cmp_fun(self.obj, other.obj, model) < 0
        def __gt__(self, other):
            return cmp_fun(self.obj, other.obj, model) > 0
        def __eq__(self, other):
            return cmp_fun(self.obj, other.obj, model) == 0
        def __le__(self, other):
            return cmp_fun(self.obj, other.obj, model) <= 0
        def __ge__(self, other):
            return cmp_fun(self.obj, other.obj, model) >= 0
        def __ne__(self, other):
            return cmp_fun(self.obj, other.obj, model) != 0
    return K

def mean_average_position():
    """ MAP = average_pos(Q) / len(Q)
    """
    pass

def discounted_cumulative_gain(ranked_list):
    """ Discounted cumulative gain sums the relevance scores, where the relevant
    scores in highly ranked documents are rated higher.

    dcg = sum(rel_i / (log_2(i + 1)))
    idcg = dcg based on ideal numbers
    ndcg = dcg / idcg

    andcg = ndcg / # queries
    """
    total_ndcg = 0
    for query in ranked_list:
        relevances = [doc[0] for doc in query[1]]
        dcg = 0
        for i, rel in enumerate(relevances, 1):
            dcg += rel / (math.log(i + 1, 2))

        idcg = 0
        for i, rel in enumerate(sorted(relevances, reverse=True), 1):
            idcg += rel / (math.log(i + 1, 2))

        total_ndcg += dcg / idcg
    return total_ndcg / len(ranked_list)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", required=True, type=validate_file,
                        help="pickled data of the model", metavar="FILE")
    parser.add_argument("-f", "--file", dest="libsvm_file", required=True, type=validate_file,
                        help="training data in libsvm format", metavar="FILE")
    args = parser.parse_args()

    model = pickle.load(open(args.model, 'rb'))
    ranked_list = import_libsvm_and_rank(model, args.libsvm_file)
    print(discounted_cumulative_gain(ranked_list))
