#!/usr/bin/env python

#File contains various definitions of features, and functions to extract it

from enum import Enum
import document_frequency
import argparse
import loader
import data_combining
import pandas as pd
import numpy as np

class Type(Enum):
    IDF = 1

class IDF_Feature:
    def __init__(self, file, column, data):
        self.file = file
        self.column = column
        corpus = document_frequency.load_corpus(self.file, self.column, data)
        vectorizer, corpus_vector = document_frequency.vectorize(corpus)
        self.vectorizer = vectorizer
        self.corpus_vector = corpus_vector

    def get_query_cols(self, query):
        transformed = self.vectorizer.transform([query])
        ctransformed = transformed.tocoo()
        self.query_cols = ctransformed.col

    def get_vectorize(self, query):
        return document_frequency.term_frequency(query, self.vectorizer, self.corpus_vector)

    def get_feature(self, id):
        return document_frequency.term_frequency_article(self.query_cols, self.corpus_vector, id)

def training_file_to_libsvm(input_file, data, output_file, verbose = False, max_rows = None):
    training_data = pd.read_json(input_file, lines = True)

    #empty output file
    open(output_file, 'w').close()

    F1 = IDF_Feature('corpus_file', 'paperAbstract', data)
    F2 = IDF_Feature('corpus_file', 'title', data)

    for index, row in training_data.iterrows():
        if verbose:
            print('Current progress: {}/{}'.format(index + 1, training_data.shape[0]), end='\r')

        query = row['query']

        F1.get_vectorize(query)
        F2.get_vectorize(query)
        F1.get_query_cols(query)
        F2.get_query_cols(query)

        for article in row['documents']:
            relevance = article['relevance']
            sha = article['doc_id']
            id = data_combining.sha_to_id(data, sha)
            if id == None:
                continue

            feature_vector = [F1.get_feature(id), F2.get_feature(id)]

            for feature in feature_vector:
                if feature == None:
                    continue

            with open(output_file, "a") as out:
                out.write("{} F1:{} F2:{}\n".format(relevance, feature_vector[0], feature_vector[1]))

    if verbose:
        print("\nFeature extraction done! Saved in libsvm format to {}".format(output_file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", dest="infile", required=True,
                        help="training file", metavar="FILE")
    parser.add_argument("-o", "--out", dest="outfile", required=True,
                        help="output file", metavar="FILE")
    args = parser.parse_args()

    data = loader.parse_files()
    training_file_to_libsvm(args.infile, data, args.outfile, verbose = True)
