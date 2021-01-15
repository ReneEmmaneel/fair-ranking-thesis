#!/usr/bin/env python

#Given a document id, we have to return a certain amount of information
#by combining the different datatables, which were extracted in the loader.py
#The file can be used on the command line:
# -python3 data-combining.py -s [sha]
# -python3 data-combining.py -i [id]
#Or the file can be used by calling the following function:
# -get_data(id_or_sha, data = None, use_sha = False)

import loader
import argparse, os
import pandas as pd
import numpy as np

def load_article_by_id(data, id):
    return data['paper_file'].loc[id, :]

def load_article_by_sha(data, sha):
    pf = data['paper_file']
    article =  pf.loc[pf['paper_sha'] == sha]
    #confim that exaclty 1 article is found
    if article.shape[0] == 1:
        return article.iloc[0]
    else:
        print('error: found ' + str(article.shape[0]) + ' articles in s2 data with given sha, expected 1.')
        return pd.Series([])

def load_s2_data(data, article):
    cf = data['corpus_file']
    s2_data = cf.loc[cf['id'] == article['paper_sha']]
    #confim that exaclty 1 datapoint is found
    if s2_data.shape[0] == 1:
        return s2_data.iloc[0]
    else:
        print('error: found ' + s2_data.shape[0] + ' articles in s2 data with given sha, expected 1.')
        return pd.Series([])

def load_authors(data, authors):
    author_list = np.array([author['ids'] for author in authors], dtype=object)
    author_int_ids = [id for id in author_list.flatten() if id is not None]
    author_int_ids = [int(id) for id in author_int_ids if type(id) is not list]
    af = data['author_metadata_file']
    author_data = af[af['corpus_author_id'].isin(author_int_ids)]
    return author_data

def get_data(id_or_sha, data = None, use_sha = False):
    #The main function for the file.
    #Given a id or sha to indentify the article, as well as the data itself,
    #return a dict in the following format:
    # {'article': pandas row containing data from the article file,
    #  'metadata': pandas row container data from the S2 Corpus file,
    #  'authors': pandas datafram containing metadata from the possibly multiple authors}
    #
    # If data is not given, load the data using loader.py
    # If use_sha is not given, assume the id is being used as indentifier
    if data == None:
        data = loader.parse_files()
    if use_sha:
        article = load_article_by_sha(data, id_or_sha)
    else:
        article = load_article_by_id(data, int(id_or_sha))

    if article.empty:
        return None

    s2_metadata = load_s2_data(data, article)

    authors = None
    if not s2_metadata.empty:
        authors = load_authors(data, s2_metadata['authors'])

    return {'article': article, 'metadata': s2_metadata, 'authors': authors}

def sha_to_id(data, sha):
    pf = data['paper_file']
    article =  pf.index[pf['paper_sha'] == sha].tolist()
    if len(article) > 0:
        return article[0]
    else:
        return None

def print_article_data(article_data):
    print('data for article \'{}\':'.format(article_data['article']['paper_title']))

    print('paper data:\n')
    print(article_data['article'])

    print('\npaper S2 Corpus data:\n')
    print(article_data['metadata'])

    print('\npaper author data:\n')
    print(article_data['authors'])

if __name__ == "__main__":
    #Give which document should be extracted
    #When no document is given, or when 2 documents are given, stop the program
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--id", dest="id", required=False,
                        help="given id of paper", metavar="INT")
    parser.add_argument("-s", "--sha", dest="sha", required=False,
                        help="given sha of paper", metavar="STRING")
    args = parser.parse_args()

    if (not args.id and not args.sha or args.id and args.sha):
        print('Wrong amount of arguments. Exactly 1 expected.')
        os._exit(0)

    #load the relevant data for given article
    if args.id:
        article_data = get_data(args.id, None, False)
    else:
        article_data = get_data(args.sha, None, True)

    print_article_data(article_data)

    #data = loader.parse_files()
