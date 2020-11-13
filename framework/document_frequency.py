#!/usr/bin/env python

#Feature extraction framework file
#Using sklearn
#usage: feature-extraction.py [-h] -q STRING
#It will return the article id with highest relevance

from sklearn.feature_extraction.text import TfidfVectorizer
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
    vectorizer = TfidfVectorizer()
    corpus_vector = vectorizer.fit_transform(corpus)

    return (vectorizer, corpus_vector)

def term_frequency(query, vectorizer, corpus_vector):
    # Get the TF-IDF scores for the query, use that to calculate
    # the relevance for each corpus_vector_row
    # TODO: currently idf score for each word in the query is just added to
    # create a feature score. This is (probably) not ideal
    transformed = vectorizer.transform([query])
    ctransformed = transformed.tocoo()
    query_cols = ctransformed.col

    feature = np.empty([corpus_vector.get_shape()[0]])

    cvector = corpus_vector.tocoo()
    for article in range(corpus_vector.get_shape()[0]):
        score = 0

        for col in query_cols:
            score += corpus_vector[article, col]

        feature[article] = score
    return feature

def term_frequency_article(query_cols, corpus_vector, id):
    try:
        score = 0
        for col in query_cols:
            score += corpus_vector[id, col]
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
    vectorizer, corpus_vector = vectorize(corpus)
    feature = term_frequency(args.query, vectorizer, corpus_vector)

    print('highest relevance is article with id: {}'.format(np.argmax(feature)))
