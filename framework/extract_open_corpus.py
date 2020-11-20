#!/usr/bin/env python

# File to extract Semantic Scolar (S2) Open Corpus to TREC format

import argparse, os
import pandas as pd
import configparser

def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

def parse_corpus_files(corpus_file):
    if corpus_file == None or (not os.path.exists(corpus_file)) or (not corpus_file.endswith('.jsonl')):
        return

    corpus_data = pd.read_json(corpus_file, lines = True)
    trec_data = corpus_data[['id', 'title', 'year', 'venue', 'inCitations', 'outCitations']]
    trec_data.rename(columns={'id': 'paper_sha', 'title': 'paper_title', 'year': 'paper_year', 'venue': 'paper_venue', 'inCitations': 'n_citations', 'outCitations': 'n_key_citations'},  inplace=True)

    trec_data['n_citations'] = trec_data.apply(lambda x: len(x['n_citations']),axis=1)
    trec_data['n_key_citations'] = trec_data.apply(lambda x: len(x['n_key_citations']),axis=1)

    return trec_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus", dest="corpus_file", required=True, type=validate_file,
                        help="corpus file", metavar="FILE")

    args = parser.parse_args()
    print(parse_corpus_files(args.corpus_file))
