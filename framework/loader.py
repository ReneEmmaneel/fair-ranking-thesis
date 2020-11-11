#!/usr/bin/env python

#Load.py:
# When called on the command line, it will read the config.cfg file
# and the given command line argument files.
# If --update flag is given, it will update the files to the config.cfg file,
# then it will load the files into panda datatables, and print them.
#
# The different arguments are:
# -p: give the paper metadata file
# -am: give the author metadata file
# -ap: give the author paper link file
# -c: give the corpus file
# -g: give the group example file
# --update: when this flag is given, update the config.cfg file with the new files
import argparse, os
import pandas as pd
import configparser

config_path = "config.cfg"

def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

def parse_arguments():
    #using argparse, parse the command line arguments for the different filenames
    #when update is true, update the config file
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--paper", dest="paper_file", required=False, type=validate_file,
                        help="paper metadata file", metavar="FILE")
    parser.add_argument("-am", "--author-metadata", dest="author_metadata_file", required=False, type=validate_file,
                        help="author metadata file", metavar="FILE")
    parser.add_argument("-ap", "--author", dest="authors_papers_file", required=False, type=validate_file,
                        help="authors for papers file", metavar="FILE")
    parser.add_argument("-c", "--corpus", dest="corpus_file", required=False, type=validate_file,
                        help="corpus file", metavar="FILE")
    parser.add_argument("-g", "--group", dest="group_file", required=False, type=validate_file,
                        help="group file", metavar="FILE")

    parser.add_argument("-u", "--update", dest="update_config", action="store_true")
    parser.add_argument("--no-update", dest="update_config", action="store_false")
    parser.set_defaults(update_config=False)

    args = parser.parse_args()

    if args.update_config:
        config = configparser.SafeConfigParser()
        config.read(config_path)

        if config.has_section('input_files'):
            cfgfile = open(config_path, 'w')

            for key in dict(config.items('input_files')):
                if key in vars(args):
                    if getattr(args, key):
                        config.set('input_files', key, str(getattr(args, key)))
                    else:
                        config.set('input_files', key, config.get('input_files', key))
            config.write(cfgfile)
            cfgfile.close()

    config = configparser.SafeConfigParser()
    config.read(config_path)
    if config.has_section('input_files'):
        config_dict = dict(config.items('input_files'))
        for arg in vars(args):
            if arg in config_dict.keys() and getattr(args, arg) is not None:
                config_dict[arg] = getattr(args, arg)

        return config_dict
    return None

def parse_files(input_dict=None):
    #given a dict with keys and filenames, extract the data from the given files
    #and return a dict with key and datatable pairs
    if input_dict == None:
        config = configparser.SafeConfigParser()
        config.read(config_path)

        if config.has_section('input_files'):
            input_dict = dict(config.items('input_files'))

    output_dict = {}
    for key in input_dict:
        try:
            file_name = input_dict[key]
            if file_name.endswith('.csv'):
                output_dict[key] = pd.read_csv(file_name)
            elif file_name.endswith('.jsonl'):
                output_dict[key] = pd.read_json(file_name, lines = True)
        except ValueError:
            print("ValueError: pandas failed to read file " + key)
            output_dict[key] = None

    return output_dict

def display_datatable_dict(datatable_dict, max_rows = 10):
    #print the given (name, pandas datatable) dicts
    pd.set_option('display.max_rows', max_rows)
    for key in datatable_dict:
        print("\n" + key + ":\n")
        print(datatable_dict[key])

if __name__ == "__main__":
    file_dict = parse_arguments()
    data = parse_files(file_dict)
    display_datatable_dict(data)
