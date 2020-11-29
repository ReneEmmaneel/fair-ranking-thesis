#!/usr/bin/env python

#File contains various definitions of features, and functions to extract it

from enum import Enum
import document_frequency, os
import argparse
import loader
import data_combining
import pandas as pd
import numpy as np
import scipy

#Different types of features, currently only inverse document frequency
class Type(Enum):
    NONE = 0
    IDF = 1
    LEN = 2
    MISC = 3

class Feature:
    def __init__(self):
        self.type = Type.NONE

    def process_query(self, query):
        pass

    def get_features(self, id):
        pass

    def get_name(self):
        return "unnamed feature"

    def get_feature_names(self):
        return [get_name()]

class IDF_Feature(Feature):
    """IDF type features.
    This feature consist of: TF, IDF, TF_IDF, BM25, LMIR
    """
    def __init__(self, file, column, data):
        self.query_dependant = True
        self.document_dependant = True

        self.type = Type.IDF
        self.file = file
        self.column = column
        corpus = document_frequency.load_corpus(self.file, self.column, data)
        vectorizer, tf_matrix, idf_vector, tfidf_matrix = document_frequency.vectorize(corpus)
        self.vectorizer = vectorizer
        self.tf_matrix = tf_matrix
        self.idf_vector = idf_vector
        self.tfidf_matrix = tfidf_matrix

        self.doc_length_vector = scipy.sparse.csr_matrix.sum(self.tf_matrix, axis=1)
        self.doc_average = self.doc_length_vector.mean()

    def get_name(self):
        return "IDF feature"

    def get_feature_names(self):
        return ['TF(q, d) in {}'.format(self.column), 'IDF(q) in {}'.format(self.column),
                'TF_IDF(q, d) in {}'.format(self.column), 'BM25 of {}'.format(self.column),
                'LMIR_ABS(q, d) in {}'.format(self.column), 'LMIR_DIR(q, d) in {}'.format(self.column),
                'LMIR_JM(q, d) in {}'.format(self.column)]

    def process_query(self, query):
        self.get_vectorize(query)
        self.get_query_cols(query)

    def get_query_cols(self, query):
        transformed = self.vectorizer.transform([query])
        ctransformed = transformed.tocoo()
        self.query_cols = ctransformed.col

    def get_vectorize(self, query):
        return document_frequency.term_frequency_inverse_document_frequency(query, self.vectorizer, self.tfidf_matrix)

    def get_features(self, id):
        tf_feature = document_frequency.term_frequency_feature(self.query_cols, self.tf_matrix, id)
        idf_feature = document_frequency.inverse_document_frequency_feature(self.query_cols, self.idf_vector)
        tfidf_feature = document_frequency.term_frequency_inverse_document_frequency_feature(self.query_cols, self.tfidf_matrix, id)
        bm25 = document_frequency.bm25_feature(self.query_cols, self.tf_matrix, self.idf_vector, self.doc_length_vector, self.doc_average, id)
        lmir_abs = document_frequency.lmir_abs_feature(self.query_cols, self.tf_matrix, self.doc_length_vector, id)
        lmir_dir = document_frequency.lmir_dir_feature(self.query_cols, self.tf_matrix, self.doc_length_vector, id)
        lmir_jm = document_frequency.lmir_jm_feature(self.query_cols, self.tf_matrix, self.doc_length_vector, id)
        return [tf_feature, idf_feature, tfidf_feature, bm25, lmir_abs, lmir_dir, lmir_jm]

class LEN_Feature(Feature):
    def __init__(self, file, column, data):
        self.query_dependant = False
        self.document_dependant = True

        self.type = Type.IDF
        self.file = file
        self.column = column
        corpus = document_frequency.load_corpus(self.file, self.column, data)
        self.lengths = [len(c) for c in corpus]

    def get_features(self, id):
        return([self.lengths[id]])

    def get_name(self):
        return "length feature"

    def get_feature_names(self):
        return ["LEN(d) of {}".format(self.column)]

def int_if_possible(str_to_int):
    try:
        return int(str_to_int)
    except ValueError:
        return None

class NUM_Feature(Feature):
    def __init__(self, file, column, data):
        self.query_dependant = False
        self.document_dependant = True

        self.type = Type.IDF
        self.file = file
        self.column = column

        file_data = data[file]
        column_data = file_data.loc[:, column]
        self.nums = [int_if_possible(d) for d in column_data]

    def get_features(self, id):
        return([self.nums[id]])

    def get_name(self):
        return "number feature"

    def get_feature_names(self):
        return ["num of {}".format(self.column)]

def training_file_to_libsvm(input_file, data, output_file, features, verbose = False, max_rows = None):
    training_data = pd.read_json(input_file, lines=True)

    #empty output file
    try:
        open(output_file, 'w').close()
    except FileNotFoundError:
        print("FileNotFoundError: Directory probably does not exist")
        return

    for index, row in training_data.iterrows():
        if verbose:
            print('Current progress: {}/{}'.format(index + 1, training_data.shape[0]), end='\r')

        query = row['query']
        qid = row['qid']

        for f in features:
            f.process_query(query)

        for article in row['documents']:
            relevance = article['relevance']
            sha = article['doc_id']
            id = data_combining.sha_to_id(data, sha)
            if id == None:
                continue

            feature_vector = []

            for index_feature, f in enumerate(features):
                new_features = f.get_features(id)
                if new_features == None:
                    print('error on line {}, feature {}'.format(index_feature, f.get_name()))
                feature_vector.extend(new_features)


            with open(output_file, "a") as out:
                out.write("{} qid:{} {}\n".format(relevance, qid, ' '.join('' if f is None else '{}:{}'.format(i, f) for i, f in enumerate(feature_vector))))

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



    features = [IDF_Feature('corpus_file', 'paperAbstract', data),
                IDF_Feature('corpus_file', 'title', data),
                IDF_Feature('corpus_file', 'venue', data),
                LEN_Feature('corpus_file', 'paperAbstract', data),
                LEN_Feature('corpus_file', 'title', data),
                LEN_Feature('corpus_file', 'venue', data),
                NUM_Feature('paper_file', 'n_citations', data)
                ]

    print('Features:')
    feature_names = [f.get_feature_names() for f in features]
    feature_names_flat = [item for sublist in feature_names for item in sublist]
    print('\n'.join('{}:\t{}'.format(i, f) for i, f in enumerate(feature_names_flat)))

    training_file_to_libsvm(args.infile, data, args.outfile, features, verbose = True)
