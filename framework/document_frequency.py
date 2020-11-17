#!/usr/bin/env python

#Feature extraction framework file
#Using sklearn
#usage: feature-extraction.py [-h] -q STRING
#It will return the article id with highest relevance

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
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
