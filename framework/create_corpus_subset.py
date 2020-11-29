# Function to create a subset of the entire corpus,

import argparse, os
import pandas as pd
import numpy as np
import requests
import time
import json
import sys

def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

def get_all_doc_id(train_file):
    training_data = pd.read_json(train_file, lines = True)

    doc_id_list = set()

    for index, row in training_data.iterrows():
        for doc in row['documents']:
            if doc['doc_id'] not in doc_id_list:
                doc_id_list.add(doc['doc_id'])

    return list(doc_id_list)

def lineclear():
    print ("\033[A                                                                                       \033[A")

def get_all_documents(doc_ids, out_file, verbose=True):
    for i, doc in enumerate(doc_ids):
        try:
            status_code, doc_json = execute_api(doc)
        except:
            if verbose:
                e = sys.exc_info()[0]
                lineclear()
                print('general error in id={} | doc={}: {}'.format(i, doc, e))
                print('Current progress: {}/{}'.format(i + 1, len(doc_ids)))
            continue

        if verbose:
            lineclear()
            print('Current progress: {}/{}'.format(i + 1, len(doc_ids)))

        #the api only allows 100 request per 5 min per IP, and gives a 403 (but the site says 429) if
        #the amount of requests exceeds it. Therefor, wait if the status_code is not 403 or 429
        start_sleep = time.time()
        while status_code == 403 or status_code == 429:
            lineclear()
            print('Current progress: {}/{} [sleeping for {} sec...] [last status code: {}]'.format(i + 1, len(doc_ids), int(time.time() - start_sleep), status_code))
            time.sleep(30)
            status_code, doc_json = execute_api(doc)

        if status_code == 200:
            with open(out_file, 'a') as outfile:
                outfile.write(doc_json)
                outfile.write('\n')
        else:
            lineclear()
            print('error with file id={} | doc={} | status_code = {}\n'.format(i, doc, status_code))

    if verbose:
        lineclear()
        print("Extracting corpus subset done! Saved in jsonl format to {}".format(out_file))


def docs_in_corpus_subset(file):
    """Given a corpus subset, return all S2 IDs
    """
    df = pd.read_json(file, lines=True)
    if not df.empty:
        return df['id'].tolist()
    return []

def execute_api(document_id):
    """Execute the API as described here:
    http://s2-public-api-prod.us-west-2.elasticbeanstalk.com/
    """
    url = 'https://api.semanticscholar.org/v1/paper/{}'.format(document_id)

    response = requests.get(url)

    if response.status_code == 200:
        response_json = response.json()

        outCitations = []
        for citation in response_json['references']:
            outCitations.append(citation['paperId'])

        inCitations = []
        for citation in response_json['citations']:
            outCitations.append(citation['paperId'])

        authors = []
        for author in response_json['authors']:
            authors.append({'name': author['name'], 'ids': [author['authorId']]})

        doc_dict = {
            'entities': [],
            'magId': '',
            'journalVolume': '',
            'journalPages': '',
            'fieldsOfStudy': response_json['fieldsOfStudy'],
            'year': response_json['year'],
            'outCitations': outCitations,
            's2Url': response_json['url'],
            's2PdfUrl': '',
            'id': document_id,
            's2id': response_json['paperId'],
            'authors': authors,
            'journalName': response_json['venue'],
            'paperAbstract': response_json['abstract'],
            'inCitations': inCitations,
            'pdfUrls': [],
            'title': response_json['title'],
            'doi': response_json['doi'],
            'sources': [],
            'doiUrl': '',
            'venue': response_json['venue']
        }

        doc_json = json.dumps(doc_dict)
        return response.status_code, doc_json
    else:
        return response.status_code, response.json()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", dest="training_file", required=True, type=validate_file, help="training file", metavar="FILE")
    parser.add_argument("-o", "--out", dest="out_file", required=True, type=validate_file, help="output file", metavar="FILE")
    args = parser.parse_args()

    if args.training_file:
        doc_id_list = get_all_doc_id(args.training_file)
        doc_id_list_corpus = docs_in_corpus_subset(args.out_file)
        doc_id_to_extract = [id for id in doc_id_list if id not in doc_id_list_corpus]

        print('Amount of total documents to be extracted: {}'.format(len(doc_id_list)))
        print('Amount of documents to be extracted: {}'.format(len(doc_id_to_extract)))
        print('')

        get_all_documents(doc_id_to_extract, args.out_file)
