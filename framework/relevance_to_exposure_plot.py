import matplotlib.pyplot as plt
import argparse, os
import sys
import pickle
import model_ranking
import loader
from training import RankSVM
import numpy as np

def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

def to_perc(num):
    return '{:.1f}%'.format(num * 100)

def plot(data):
    ind = np.arange(2)
    width = 0.4
    bottom = (0,0)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)


    for key in sorted(data):
        ax.bar(ind, data[key], width, bottom = bottom, label='Group {}'.format(key))

        for i in [0,1]:
            plt.text(ind[i], (bottom[i] + data[key][i] / 2), to_perc(data[key][i]),
                     horizontalalignment='center', verticalalignment='center')

        bottom = tuple(map(sum, zip(bottom, data[key])))

    plt.xticks(ind, ('relevance', 'exposure'))
    plt.ylabel('Percentage')
    plt.title('Exposure and relevance for each group.\nGroups are determined by amount of publications with at least 10 citations.')

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles[::-1], labels[::-1])

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", required=True, type=validate_file,
                        help="pickled data of the model", metavar="FILE")
    parser.add_argument("-f", "--file", dest="libsvm_file", required=True, type=validate_file,
                        help="training data in libsvm format", metavar="FILE")
    parser.add_argument("-g", "--group", dest="group", required=True, type=validate_file,
                        help="group definition", metavar="FILE")
    args = parser.parse_args()

    data = loader.parse_files()
    model = pickle.load(open(args.model, 'rb'))
    ranked_list = model_ranking.import_libsvm_and_rank(model, args.libsvm_file)
    data = model_ranking.fairness_ranking(data, ranked_list, args.group)[1]

    plot(data)
