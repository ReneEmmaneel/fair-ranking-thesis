#!/usr/bin/env python

#Given a document id, we have to return a certain amount of information
#by combining the different datatables, which were extracted in the loader.py

import loader
import argparse, os
import pandas as pd

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

    #find the article
    data = loader.parse_files()

    print("Data loaded!")

    if args.id and not args.sha:
        article = load_article_by_id(data, int(args.id))
    elif not args.id and args.sha:
        article = load_article_by_sha(data, args.sha)

    if article.empty:
        os._exit(0)

    #find the S2 data
    s2_metadata = load_s2_data(data, article)
    print(s2_metadata)

    #data = loader.parse_files()
