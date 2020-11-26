#!/usr/bin/env python

#Feature extraction framework file
#Using sklearn
#usage: feature-extraction.py [-h] -q STRING
#It will return the article id with highest relevance

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import math
import pandas as pd
import argparse, os
import numpy as np
import loader


def load_corpus(file, column, data = None):
    if data == None:
        data = loader.parse_files()

    file_data = data[file]
    column_data = file_data.loc[:, column]

    return column_data

def vectorize(corpus):
    #Generate tf-vector, idf-vector, idf-matrix and return the values
    #Also returns the vecgtorizer object
    vectorizer = TfidfVectorizer()
    fit = vectorizer.fit(corpus)
    tfidf_matrix = vectorizer.transform(corpus)

    tf_vectorizer = CountVectorizer()
    tf_matrix = tf_vectorizer.fit_transform(corpus)

    return (vectorizer, tf_matrix, fit.idf_, tfidf_matrix)

def term_frequency_inverse_document_frequency(query, vectorizer, tfidf_matrix):
    # Get the TF-IDF scores for the query, use that to calculate
    # the relevance for each corpus_vector_row
    transformed = vectorizer.transform([query])
    ctransformed = transformed.tocoo()
    query_cols = ctransformed.col

    feature = np.empty([tfidf_matrix.get_shape()[0]])

    cvector = tfidf_matrix.tocoo()
    for article in range(tfidf_matrix.get_shape()[0]):
        score = 0

        for col in query_cols:
            score += tfidf_matrix[article, col]

        feature[article] = score
    return feature

def term_frequency_feature(query_cols, tf_matrix, id):
    try:
        score = 0
        for col in query_cols:
            score += tf_matrix[id, col]
        return score
    except IndexError:
        return None

def inverse_document_frequency_feature(query_cols, idf_vector):
    try:
        score = 0
        for col in query_cols:
            score += idf_vector[col]
        return score
    except IndexError:
        return None

def term_frequency_inverse_document_frequency_feature(query_cols, tfidf_matrix, id):
    try:
        score = 0
        for col in query_cols:
            score += tfidf_matrix[id, col]
        return score
    except IndexError:
        return None

def bm25_feature(query_cols, tf_matrix, idf_vector, doc_length_vector, doc_average, id, k = 1.2, b = 0.75):
    try:
        d = doc_length_vector[id, 0]
        score = 0
        for col in query_cols:
            idf = idf_vector[col]
            tf = tf_matrix[id, col]
            score += idf * (tf * (k + 1.) / (tf + k * (1. - b + b * d / doc_average)))
        return score
    except IndexError:
        return None

def lmir_jm_feature(query_cols, tf_matrix, doc_length_vector, id, l = 0.1):
    """the Jelinek Mercer LMIR feature

    P(t | d) = (1 - lambda) (TF(t, d) / LEN(d)) + lambda P(t | C)"""
    try:
        d = doc_length_vector[id, 0]

        tot_tf_count_vector = tf_matrix.sum(axis=0)
        tot_tf_count = doc_length_vector.sum()

        score = 0
        for col in query_cols:
            if d == 0:
                continue


            tf = tf_matrix[id, col]
            prob_corpus = tot_tf_count_vector[0, col] / tot_tf_count

            prob_term = (1 - l) * tf / d + l * prob_corpus
            score -= math.log(prob_term)

        return None if math.isnan(score) else score
    except IndexError:
        return None

def lmir_dir_feature(query_cols, tf_matrix, doc_length_vector, id, mu = 2000):
    """the Dirichlet LMIR feature

    P(t | d) = (TF(t) + mu * P(w | C)) / (LEN(d) + mu)
    """
    try:
        d = doc_length_vector[id, 0]

        tot_tf_count_vector = tf_matrix.sum(axis=0)
        tot_tf_count = doc_length_vector.sum()

        score = 0
        for col in query_cols:
            if d == 0:
                continue

            tf = tf_matrix[id, col]
            prob_corpus = tot_tf_count_vector[0, col] / tot_tf_count

            prob_term = (tf + mu * prob_corpus) / (d + mu)
            score -= math.log(prob_term)

        return None if math.isnan(score) else score
    except IndexError:
        return None

def lmir_abs_feature(query_cols, tf_matrix, doc_length_vector, id, delta = 0.7):
    """the Dirichlet LMIR feature

    P(t | d) = (max(TF - delta, 0) / doc_len) + delta * unique_terms / d * P(t | C)
    """
    try:
        d = doc_length_vector[id, 0]

        tot_tf_count_vector = tf_matrix.sum(axis=0)
        tot_tf_count = doc_length_vector.sum()

        score = 0
        for col in query_cols:
            if d == 0:
                continue

            tf = tf_matrix[id, col]
            prob_corpus = tot_tf_count_vector[0, col] / tot_tf_count

            unique_terms = tf_matrix[:,col].count_nonzero()

            prob_term = max(tf - delta,0) / d + delta * unique_terms / d * prob_corpus
            score -= math.log(prob_term)

        return None if math.isnan(score) else score
    except IndexError:
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", dest="query", required=True,
                        help="query", metavar="STRING")
    args = parser.parse_args()
    print("Query: {}".format(args.query))

    corpus = load_corpus('corpus_file', 'paperAbstract')
    vectorizer, tf_matrix, idf_vecotr, tfidf_matrix = vectorize(corpus)
    feature = term_frequency_inverse_document_frequency(args.query, vectorizer, tfidf_matrix)

    print('highest relevance is article with id: {}'.format(np.argmax(feature)))
