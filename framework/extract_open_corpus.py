#!/usr/bin/env python

# File to extract Semantic Scolar (S2) Open Corpus to TREC format
# See thesis for formal full definition of TREC format
#
# usage: extract_open_corpus.py [-h] -c FILE -s FOLDER
# example:
# python3 extract_open_corpus.py -c ../../thesis/data/corpus-subset-for-queries.jsonl -s ../../thesis/extract_open_corpus_test/

import argparse, os
import pandas as pd
import numpy as np
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

    pd.set_option('mode.chained_assignment', None)
    corpus_data = pd.read_json(corpus_file, lines = True)
    trec_paper_data = corpus_data[['id', 'title', 'year', 'venue', 'authors', 'inCitations', 'outCitations']]
    trec_paper_data.rename(columns={'id': 'paper_sha', 'title': 'paper_title', 'year': 'paper_year', 'venue': 'paper_venue', 'outCitations': 'n_citations', 'inCitations': 'n_key_citations'},  inplace=True)

    trec_paper_data['n_citations'] = trec_paper_data.apply(lambda x: len(x['n_citations']),axis=1)
    trec_paper_data['n_key_citations'] = trec_paper_data.apply(lambda x: len(x['n_key_citations']),axis=1)

    author_dict = {}
    trec_author_list = []
    trec_linking_list = []

    # Loop through paper data to extract all authors
    for index, row in trec_paper_data.iterrows():
        for position, author in enumerate(row['authors'], 1):
            for id in author['ids']:
                trec_linking_list.append({'paper_sha': row['paper_sha'], 'corpus_author_id': id, 'position': position})
                if id in author_dict:
                    author_dict[id]['papers'].append(row['n_key_citations'])
                else:
                    author_dict[id] = {'papers': [row['n_key_citations']], 'name': author['name']}

    trec_linking_data = pd.DataFrame(trec_linking_list)

    # Loop through all authors and calculate i10/h_index
    for id, author in author_dict.items():
        paper_citations = [int(x) for x in author['papers']]
        paper_citations.sort()
        citation_count = sum(paper_citations)
        paper_count = len(paper_citations)
        i10 = sum([1 if x >= 10 else 0 for x in paper_citations])
        h_index = 0
        for i in range(1,len(paper_citations)+1)[::-1]:
            if paper_citations[-i] >= i:
                h_index = i
                break
        h_class = 'H' if h_index >= 10 else 'L'
        trec_author_list.append({'corpus_author_id': id,'name': author['name'], 'num_citations': citation_count, 'num_papers': paper_count, 'i10': i10, 'h_index': h_index, 'h_class': h_class})

    trec_author_data = pd.DataFrame(trec_author_list)

    trec_paper_data = trec_paper_data.drop('authors', 1)

    return trec_paper_data, trec_author_data, trec_linking_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus", dest="corpus_file", required=True, type=validate_file,
                        help="corpus file", metavar="FILE")
    parser.add_argument("-s", "--save", dest="save_folder", required=True, type=validate_file,
                        help="folder to save to", metavar="FOLDER")

    args = parser.parse_args()

    trec_paper_data, trec_author_data, trec_linking_data = parse_corpus_files(args.corpus_file)

    trec_paper_data.to_csv(index=False, path_or_buf=os.path.join(args.save_folder, 'corpus-subset-for-queries.papers.csv'))
    trec_author_data.to_csv(index=False, path_or_buf=os.path.join(args.save_folder, 'corpus-subset-for-queries.authors.csv'))
    trec_linking_data.to_csv(index=False, path_or_buf=os.path.join(args.save_folder, 'corpus-subset-for-queries.paper_authors.csv'))
