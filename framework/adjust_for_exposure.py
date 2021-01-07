import model_ranking
import os, argparse
import loader
import pickle
import svm_training
from svm_training import RankSVM
import data_combining
import pandas as pd
import math
import numpy as np

def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

def adjust_for_exposure(data, ranked_list, group_file, model):
    for iter, query in enumerate(ranked_list):
        print('Current progress: {}/{}'.format(iter + 1, len(ranked_list)), end='\r')

        top_document = query[1][0][2]
        max_difference = model.predict_single_value(top_document, query[1][len(query[1]) - 1][2])[0]

        curr_fairness = fairness_ranking(data, query, group_file)[0]
        curr_relevance = relevance_ranking(data, query, model, top_document, max_difference)

        #swap adjacent ranked documents
        for i in range(len(query[1]) - 1):
            temp_list = query[1].copy()
            temp_list[i], temp_list[i + 1] = temp_list[i + 1], temp_list[i]
            temp_fairness = fairness_ranking(data, [query[0], temp_list], group_file)[0]
            temp_relevance = relevance_ranking(data, [query[0], temp_list], model, top_document, max_difference)

            #if the swap produces fairer ranking for not much relevance decrease, swap permanently
            delta_fairness = curr_fairness - temp_fairness
            delta_relevance = temp_relevance - curr_relevance
            if (delta_fairness > 0 and delta_fairness > delta_relevance):
                curr_fairness = temp_fairness
                curr_relevance = temp_relevance
                query[1] = temp_list.copy()

    return ranked_list

def fairness_ranking(data, query, group_file, gamma=0.5, stop_prob=0.7, verbose=False):
    """Fairness ranking is the algorithm implemented in the Fair Ranking track,
    which measures first both the exposure and relevance for each author,
    then it measures the group exposure and relevance,
    after which the fair exposure number can be calculated.
    This function only calculates the fairness of a single query.
    """
    author_exposure = {}
    author_relevance = {}

    #calculate the exposure and relevance for each author
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
        try:
            group_relevance = relevance / total_group_relevance
        except:
            group_relevance = 0

        try:
            group_exposure = group_exposure_sum[group] / total_group_exposure
        except:
            group_exposure = 0

        group_exposure_and_relevance[group] = (group_relevance, group_exposure)
        if verbose:
            print('{}: rel {} | exp {}'.format(group, group_relevance, group_exposure))

        squared_sum += (group_exposure - group_relevance) ** 2

    if verbose:
        print('fair exposure: {}'.format(math.sqrt(squared_sum)))

    return (math.sqrt(squared_sum), group_exposure_and_relevance)

def relevance_ranking(data, query, model, top_document, max_difference, gamma=0.5, stop_prob=0.7, verbose=False):
    """We calculate the relevance ranking using the exposure each document in a
    ranking gets by the stop probability of the document.
    The final result is the average utility of each ranking.
    This relevance ranking is only for a single query, and is based on the
    model calculated by the training.
    """
    total_relevance = 0
    exposure = 1.0

    min_difference = max_difference * -1

    for doc in query[1]:

        relevance = (model.predict_single_value(doc[2], top_document)[0] - min_difference) / (max_difference - min_difference) * 2

        total_relevance += exposure * relevance * stop_prob

        exposure *= gamma
        exposure *= (1 - stop_prob * relevance)
    return total_relevance

if __name__ == "__main__":
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
    ranked_list = model_ranking.import_libsvm_and_rank(model, args.libsvm_file, args.linker_file)

    # print(discounted_cumulative_gain(ranked_list))
    print('fairness: {}'.format(model_ranking.fairness_ranking(data, ranked_list, args.group, verbose=False)[0]))
    print('utility: {}'.format(model_ranking.relevance_ranking(data, ranked_list)))

    adjusted_list = adjust_for_exposure(data, ranked_list, args.group, model)
    print('fairness: {}'.format(model_ranking.fairness_ranking(data, adjusted_list, args.group, verbose=False)[0]))
    print('utility: {}'.format(model_ranking.relevance_ranking(data, adjusted_list)))
