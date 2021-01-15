#!/usr/bin/env python

# File containing various ranking functions for ranking models.
# It contains both functions for ranking and for fair ranking.

# The functions implemented:
# -Discounted Cumulative Gain
# -Fair ranking metrics as described in the TREC Fair Ranking

import numpy as np
import argparse, os
import pandas as pd
import svm_training
from svm_training import RankSVM
from sklearn import datasets
import pickle
import math
import data_combining
import loader

def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

def import_libsvm_and_rank(model, in_file, linker_file):
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

    linker_dataframe = pd.read_csv(linker_file)

    # step 1, make a ranked_list with qid, rel_vector, feature_vec
    ranked_list = []
    for i, x in enumerate(X):
        #if qid already exists, add to the list

        added = False
        for query in ranked_list:
            if query[0] == qid[i]:
                query[1].append((y[i], int(linker_dataframe['document_id'].iloc[i]), x))
                added = True
                break
        #else add the qid
        if not added:
            ranked_list.append([qid[i], [(y[i], int(linker_dataframe['document_id'].iloc[i]), x)]])

    #step 2, reranking; for every query rank the feature vectors using the given model
    for i, query in enumerate(ranked_list):
        query[1] = sorted(query[1], key=cmp_to_key(sort_documents, model))

    return ranked_list

def sort_documents(a,b, model):
    return model.predict_single(b[2], a[2])

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

def relevance_ranking(data, ranked_list, gamma=0.5, stop_prob=0.7):
    """We calculate the relevance ranking using the exposure each document in a
    ranking gets by the stop probability of the document.
    The final result is the average utility of each ranking.
    """
    total_relevance = 0
    for query in ranked_list:
        exposure = 1.0
        for doc in query[1]:
            relevance = doc[0]

            total_relevance += exposure * relevance * stop_prob

            exposure *= gamma
            exposure *= (1 - stop_prob * relevance)
    return total_relevance / len(ranked_list)

def fairness_ranking(data, ranked_list, group_file, gamma=0.5, stop_prob=0.7, verbose=False):
    """Fairness ranking is the algorithm implemented in the Fair Ranking track,
    which measures first both the exposure and relevance for each author,
    then it measures the group exposure and relevance,
    after which the fair exposure number can be calculated.
    """
    author_exposure = {}
    author_relevance = {}

    #calculate the exposure and relevance for each author
    for query in ranked_list:
        exposure = 1.0
        for doc in query[1]:
            relevance = doc[0]

            authors_id = data_combining.get_data(doc[1], data = data)['authors']['corpus_author_id']

            for a in authors_id:
                if int(a) not in author_exposure:
                    author_exposure[int(a)] = exposure
                else:
                    author_exposure[int(a)] += exposure

                if int(a) not in author_relevance:
                    author_relevance[int(a)] = stop_prob * relevance
                else:
                    author_relevance[int(a)] += stop_prob * relevance

            exposure *= gamma
            exposure *= (1 - stop_prob * relevance)

    #calculate group relevance and exposure
    group_exposure_sum = {}
    group_relevance_sum = {}
    group = pd.read_csv(group_file)
    for index, row in group.iterrows():
        curr_author_exposure = author_exposure[row['author_id']] if row['author_id'] in author_exposure else 0
        curr_author_relevance = author_relevance[row['author_id']] if row['author_id'] in author_relevance else 0


        if row['gid'] not in group_exposure_sum:
            group_exposure_sum[row['gid']] = curr_author_exposure
        else:
            group_exposure_sum[row['gid']] += curr_author_exposure

        if row['gid'] not in group_relevance_sum:
            group_relevance_sum[row['gid']] = curr_author_relevance
        else:
            group_relevance_sum[row['gid']] += curr_author_relevance

    total_group_relevance = sum(group_relevance_sum.values())
    total_group_exposure = sum(group_exposure_sum.values())

    squared_sum = 0

    group_exposure_and_relevance = {}

    for group, relevance in group_relevance_sum.items():
        group_relevance = relevance / total_group_relevance
        group_exposure = group_exposure_sum[group] / total_group_exposure

        group_exposure_and_relevance[group] = (group_relevance, group_exposure)
        if verbose:
            print('{}: rel {} | exp {}'.format(group, group_relevance, group_exposure))

        squared_sum += (group_exposure - group_relevance) ** 2

    if verbose:
        print('fair exposure: {}'.format(math.sqrt(squared_sum)))

    return (math.sqrt(squared_sum), group_exposure_and_relevance)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", required=True, type=validate_file,
                        help="pickled data of the model", metavar="FILE")
    parser.add_argument("-f", "--file", dest="libsvm_file", required=True, type=validate_file,
                        help="training data in libsvm format", metavar="FILE")
    parser.add_argument("-l", "--linker", dest="linker_file", required=True, type=validate_file,
                        help="linker file for the libsvm file", metavar="FILE")
    parser.add_argument("-g", "--group", dest="group", required=True, type=validate_file,
                        help="group definition", metavar="FILE")
    args = parser.parse_args()

    data = loader.parse_files()
    model = pickle.load(open(args.model, 'rb'))
    ranked_list = import_libsvm_and_rank(model, args.libsvm_file, args.linker_file)

    # print(discounted_cumulative_gain(ranked_list))
    print('fairness: {}'.format(fairness_ranking(data, ranked_list, args.group, verbose=False)[0]))
    print('utility: {}'.format(relevance_ranking(data, ranked_list)))
