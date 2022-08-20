import gc
import glob
import hashlib
import itertools
import string
import zhon
import json
import os
import random
import pickle
import re
import math
import subprocess
import csv
import shutil
import scipy.sparse as sp
from collections import Counter
from os.path import join as pjoin
from zhon.hanzi import punctuation
import torch
import collections
from multiprocess import Pool
from nltk.corpus import stopwords
from others.logging import logger
from others.tokenization import BertTokenizer
from transformers import RobertaTokenizer, AutoTokenizer
from pytorch_transformers import XLNetTokenizer

from others.utils import clean
from prepro.utils import _get_word_ngrams

import xml.etree.ElementTree as ET
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm

nyt_remove_words = ["photo", "graph", "chart", "map", "table", "drawing"]


def recover_from_corenlp(s):
    s = re.sub(r' \'{\w}', '\'\g<1>', s)
    s = re.sub(r'\'\' {\w}', '\'\'\g<1>', s)


def clean_json(json_dict):
    #
    # how about bib? they also indicate what the paper is about in general
    #
    try:
        title = json_dict['metadata']['title']
    except KeyError:
        title = 'NA'
    text = ''
    for p in json_dict['body_text']:
        p_text = p['text']
        if p['section'] == 'Pre-publication history':
            continue
        # remove references and citations from text
        citations = [*p['cite_spans'],*p['ref_spans']]
        if len(citations):
            for citation in citations:
                cite_span_len = citation['end']-citation['start']
                cite_span_replace = ' '*cite_span_len
                p_text  = p_text[:citation['start']] + cite_span_replace  + p_text[citation['end']:]
        # do other cleaning of text 
        p_text = p_text.strip()
        p_text = re.sub('\[[\d\s,]+?\]', '', p_text) # matches references e.g. [12]
        p_text = re.sub('\(Table \d+?\)', '', p_text) # matches table references e.g. (Table 1)
        p_text = re.sub('\(Fig. \d+?\)', '', p_text) # matches fig references e.g. (Fig. 1)
        p_text = re.sub('(?<=[0-9]),(?=[0-9])', '', p_text) # matches numbers seperated by commas
        p_text = re.sub('[^\x00-\x7f]+',r'', p_text) # strips non ascii
        p_text = re.sub('[\<\>]',r' ', p_text) # strips  <> tokens which are not compatable StanfordNLPtokenizer
        p_text = re.sub('(\([0-9]+\))(?= [0-9]+)',' ',p_text) # removes numbers in brackets followed by another number are not compatable StanfordNLPtokenizer
        p_text = re.sub('\n',' ',p_text) # replaces line break with full stop
        p_text = re.sub('\r',' ',p_text) # replaces line break with full stop
        p_text = re.sub(' +',' ',p_text) # removes multipe blank spaces. 
        p_text = re.sub('(?<=[0-9])( +)(?=[0-9])', '', p_text) # matches numbers seperated by space and combines
        p_text = re.sub('(?<=\.)( +)(?=\.)', '', p_text) # matches several full stops with one or more spaces in between and removes spaces
        text += '{:s}\n'.format(p_text)

    return {'title': title, 'text': text}

def load_json(f_main, f_abs):
    with open(f_main, 'r') as f:
        json_main = json.load(f)
    with open(f_abs, 'r') as f:
        json_abs = json.load(f)

    src_sent_tokens = [
        list(t['word'].lower() for t in sent['tokens'])
        for sent in json_main['sentences']]
    if not src_sent_tokens:
        return None, None
    else:
        tgt_sent_tokens = [list(t['word'].lower() for t in sent['tokens'])
        for sent in json_abs['sentences']]
        src = [clean(' '.join(tokens)).split() for tokens in src_sent_tokens]
        tgt = [clean(' '.join(tokens)).split() for tokens in tgt_sent_tokens]
        return src, tgt

def load_xml(p):
    tree = ET.parse(p)
    root = tree.getroot()
    title, byline, abs, paras = [], [], [], []
    title_node = list(root.iter('hedline'))
    if (len(title_node) > 0):
        try:
            title = [p.text.lower().split() for p in list(title_node[0].iter('hl1'))][0]
        except:
            print(p)

    else:
        return None, None
    byline_node = list(root.iter('byline'))
    byline_node = [n for n in byline_node if n.attrib['class'] == 'normalized_byline']
    if (len(byline_node) > 0):
        byline = byline_node[0].text.lower().split()
    abs_node = list(root.iter('abstract'))
    if (len(abs_node) > 0):
        try:
            abs = [p.text.lower().split() for p in list(abs_node[0].iter('p'))][0]
        except:
            print(p)

    else:
        return None, None
    abs = ' '.join(abs).split(';')
    abs[-1] = abs[-1].replace('(m)', '')
    abs[-1] = abs[-1].replace('(s)', '')

    for ww in nyt_remove_words:
        abs[-1] = abs[-1].replace('(' + ww + ')', '')
    abs = [p.split() for p in abs]
    abs = [p for p in abs if len(p) > 2]

    for doc_node in root.iter('block'):
        att = doc_node.get('class')
        # if(att == 'abstract'):
        #     abs = [p.text for p in list(f.iter('p'))]
        if (att == 'full_text'):
            paras = [p.text.lower().split() for p in list(doc_node.iter('p'))]
            break
    if (len(paras) > 0):
        if (len(byline) > 0):
            paras = [title + ['[unused3]'] + byline + ['[unused4]']] + paras
        else:
            paras = [title + ['[unused3]']] + paras

        return paras, abs
    else:
        return None, None

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def tokenize_allenai_datasets(args):
    root_data_dir = os.path.abspath(args.raw_path)
    tokenized_data_dir = os.path.abspath(args.save_path)
    meta_path = os.path.join(root_data_dir, 'metadata.csv')
    new_meta_path = os.path.join(root_data_dir, 'PMC.csv')
    pmc_dir = os.path.join(root_data_dir, 'document_parses', 'pmc_json')
    txt_dir = os.path.join(root_data_dir, 'document_parses', 'txt_json')
    
    files_count_real = 0
    no_path_counter = 0

    # make directories for saving data if they don't already exist
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    if not os.path.exists(tokenized_data_dir):
        os.makedirs(tokenized_data_dir)
    
    print('... Loading PMC data from {}'.format(pmc_dir))

    # read in csv containing metadata about files
    df = pd.read_csv(meta_path, sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
    print('Number of files before removing papers without abstract: {}'.format(df.shape[0]))
    
    # skip papers without abstract
    df = df[~pd.isnull(df.abstract)]
    print('Number of files after removing papers without abstract: {}'.format(df.shape[0]))
    
    # drop duplicates
    df['title_lower'] = df.title.str.lower()
    df= df.drop_duplicates(subset='title_lower').drop(columns='title_lower')
    len_before = df.shape[0]
    print('Number of files once articles deduplicated: \t{}'.format(len_before)) # 56341 
    
    start = time.time()
    print('... (1) Processing pubmed files into readable .txt format for tokenizer into path: {}...'.format(txt_dir))
    
    write_head = False

    # write out new csv containing files we use in our dataset
    with open(new_meta_path, 'w') as f:
        w = csv.writer(f)
        for i,row in tqdm(df.iterrows(),total=df.shape[0]):
                
            # read in pubmed file if available
            pid = row['pmcid']
            try:
                pubtime = row['publish_time']
            except KeyError: 
                pubtime = row['year']
            # pubtime = datetime.strptime(row['publish_time'], '%Y-%m-%d').timestamp()
            ppath = os.path.join(pmc_dir, '{}.xml.json'.format(pid))
            if not os.path.isfile(ppath):
                no_path_counter +=1
                continue
            with open(ppath, 'r') as fi:
                json_dict = json.load(fi)
            
            # preprocess / clean file
            cleaned_dict = clean_json(json_dict)
            tpath = os.path.join(txt_dir, '{}-{}.txt'.format(pubtime, pid))
            tpath_abs = os.path.join(txt_dir, '{}-{}.abs.txt'.format(pubtime, pid))
            
            # write out main text and abstract 
            with open(tpath, 'w') as fil:
                fil.write(cleaned_dict['text'])
            with open(tpath_abs, 'w') as fil:
                fil.write(row['abstract'])
            files_count_real += 1
            
            # write csv row
            cleaned_dict['abstract'] = row['abstract']
            if cleaned_dict['title'] == 'NA':
                cleaned_dict['title'] = row['title']
            if not write_head:
                w.writerow(cleaned_dict.keys())
                write_head = True       
            w.writerow(cleaned_dict.values())

    end = time.time()
    print('Real count for files with abstract: {} ({}%)'.format(files_count_real,files_count_real / len_before * 100))
    print('... Ending (1), time elapsed {}'.format(end - start))

    print("Preparing to tokenize %s to %s..." % (root_data_dir, tokenized_data_dir))
    num_files_to_tokenize = 0
    # make IO list file
    print("Making list of files to tokenize...")
    with open('mapping_for_corenlp.txt', 'w') as fi:
        for fname in os.listdir(txt_dir):
            fpath = os.path.join(txt_dir, fname)
            fi.write('{}\n'.format(fpath))
            num_files_to_tokenize+=1

    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tokenized_data_dir]

    print("Tokenizing %i files in %s and saving in %s..." % (num_files_to_tokenize, txt_dir, tokenized_data_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")
    

    # Check that the tokenized data directory contains the same number of files as the original directory
    num_orig = len(os.listdir(txt_dir))
    num_tokenized = len(os.listdir(tokenized_data_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized data directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_data_dir, num_tokenized, root_data_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (root_data_dir, tokenized_data_dir))
    shutil.rmtree(txt_dir)

def clean_abstract(text_array):
    abstract = ''
    for sentence in text_array:
        sentence = sentence.replace("<S>","")
        sentence = sentence.replace("</S>","")
        abstract += sentence
    return abstract


def clean_text(doc):
    text = ''
    for sentence in doc:
            # do other cleaning of text 
            sentence = sentence.strip()
            sentence = re.sub('\[[\d\s\,]+?\]', '', sentence) # matches references e.g. [12]
            sentence = re.sub('\(table \d+?\)', '', sentence) # matches table references e.g. (Table 1)
            sentence = re.sub('\(fig. \d+?\)', '', sentence) # matches fig references e.g. (Fig. 1)
            sentence = re.sub('[^\x00-\x7f]+',r'', sentence) # strips non ascii
            sentence = re.sub('[\<\>]',r' ', sentence) # strips  <> tokens which are not compatable StanfordNLPtokenizer
            sentence = re.sub('(\([0-9]+\))(?= [0-9]+)',' ',sentence) # removes numbers in brackets followed by another number are not compatable StanfordNLPtokenizer
            sentence = re.sub('\n',' ',sentence) # replaces line break with full stop
            sentence = re.sub('\r',' ',sentence) # replaces line break with full stop
            sentence = re.sub(' +',' ',sentence) # removes multipe blank spaces. 
            sentence = re.sub('(?<=[0-9])( +)(?=[0-9])', '', sentence) # matches numbers seperated by space and combines
            sentence = re.sub('(?<=\.)( +)(?=\.)', '', sentence) # matches several full stops with one or more spaces in between and removes spaces
            #print(sentence)
            text += sentence
    
    return text

def tokenize_pubmed_dataset(args):
   
    root_data_dir = os.path.abspath(args.raw_path)
    
    dirs = ['test_pubmed','train_pubmed','val_pubmed']
    for idx, dir in enumerate(dirs):
        files_count_real = 0
        tokenized_data_dir = os.path.join(os.path.abspath(args.save_path),dir)
        source_txt_file = os.path.join(root_data_dir, '{}.jsonl'.format(dir))
        txt_dir = os.path.join(root_data_dir, 'txt_json', dir)

        # make directories for saving data if they don't already exist
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)
        if not os.path.exists(tokenized_data_dir):
            os.makedirs(tokenized_data_dir)
        
        print('... Loading PMC data from {}'.format(root_data_dir))

        # read in txt file with raw data
        df = pd.read_json(path_or_buf=source_txt_file, lines=True)
        len_before = df.shape[0]
        
        start = time.time()
        print('... (1) Processing pubmed files into readable .txt format for tokenizer into path: {}...'.format(txt_dir))
        
        # write out new csv containing files we use in our dataset
        pid = 0
        labels = []
        for i,row in tqdm(df.iterrows(),total=df.shape[0]):
                
            # read in pubmed file if available
        
            # preprocess / clean file
            try:
               label =  row['label']
            except KeyError:
                label = []
            cleaned_text = row['text']
            tpath = os.path.join(txt_dir, '{}.txt'.format(pid))
            tpath_abs = os.path.join(txt_dir, '{}.abs.txt'.format(pid))
            # preprocess/ clean abstract
            abstract = row['summary']
            #print(row)
            labels.append(label)
            #print('text:', cleaned_text)
            #print('summary:', abstract)
        
            # write out main text and abstract 
            with open(tpath, 'w') as fil:
                fil.write(cleaned_text)
            with open(tpath_abs, 'w') as fil:
                fil.write(abstract)
            files_count_real += 1
            pid += 1

        #pic_path = os.path.join(args.save_path, '{}.pkl'.format(dirs_rename[idx]))
        #with open(pic_path, 'wb') as f:
        #    pickle.dump(labels, f)

        end = time.time()
    
        print('Real count for files with abstract: {} ({}%)'.format(files_count_real,files_count_real / len_before * 100))
        print('... Ending (1), time elapsed {}'.format(end - start))

        print("Preparing to tokenize %s to %s..." % (root_data_dir, tokenized_data_dir))
        num_files_to_tokenize = 0
        # make IO list file
        print("Making list of files to tokenize...")
        with open('mapping_for_corenlp.txt', 'w') as fi:
            for fname in os.listdir(txt_dir):
                fpath = os.path.join(txt_dir, fname)
                fi.write('{}\n'.format(fpath))
                num_files_to_tokenize+=1

        command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
                '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
                'json', '-outputDirectory', tokenized_data_dir]

        print("Tokenizing %i files in %s and saving in %s..." % (num_files_to_tokenize, txt_dir, tokenized_data_dir))
        subprocess.call(command)
        print("Stanford CoreNLP Tokenizer has finished.")
        os.remove("mapping_for_corenlp.txt")
        

        # Check that the tokenized data directory contains the same number of files as the original directory
        num_orig = len(os.listdir(txt_dir))
        num_tokenized = len(os.listdir(tokenized_data_dir))
        if num_orig != num_tokenized:
            raise Exception(
                "The tokenized data directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                    tokenized_data_dir, num_tokenized, root_data_dir, num_orig))
        print("Successfully finished tokenizing %s to %s.\n" % (root_data_dir, tokenized_data_dir))
        shutil.rmtree(txt_dir)


def tokenize_pubmed_long_dataset(args):
    root_data_dir = os.path.abspath(args.raw_path)

    dirs = ['test', 'train', 'val']

    for dir in dirs:
        files_count_real = 0
        tokenized_data_dir = os.path.join(os.path.abspath(args.save_path), dir)
        source_txt_file = os.path.join(root_data_dir, '{}.txt'.format(dir))
        txt_dir = os.path.join(root_data_dir, 'txt_json', dir)

        # make directories for saving data if they don't already exist
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)
        if not os.path.exists(tokenized_data_dir):
            os.makedirs(tokenized_data_dir)

        print('... Loading PMC data from {}'.format(root_data_dir))

        # read in txt file with raw data
        df = pd.read_json(path_or_buf=source_txt_file, lines=True)
        len_before = df.shape[0]

        start = time.time()
        print(
            '... (1) Processing pubmed files into readable .txt format for tokenizer into path: {}...'.format(txt_dir))

        # write out new csv containing files we use in our dataset

        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            # read in pubmed file if available
            pid = row['article_id']

            # preprocess / clean file
            cleaned_text = clean_text(row['article_text'])
            tpath = os.path.join(txt_dir, '{}.txt'.format(pid))
            tpath_abs = os.path.join(txt_dir, '{}.abs.txt'.format(pid))

            # preprocess/ clean abstract
            abstract = clean_abstract(row['abstract_text'])

            # write out main text and abstract
            with open(tpath, 'w') as fil:
                fil.write(cleaned_text)
            with open(tpath_abs, 'w') as fil:
                fil.write(abstract)
            files_count_real += 1

        end = time.time()
        print('Real count for files with abstract: {} ({}%)'.format(files_count_real,
                                                                    files_count_real / len_before * 100))
        print('... Ending (1), time elapsed {}'.format(end - start))

        print("Preparing to tokenize %s to %s..." % (root_data_dir, tokenized_data_dir))
        num_files_to_tokenize = 0
        # make IO list file
        print("Making list of files to tokenize...")
        with open('mapping_for_corenlp.txt', 'w') as fi:
            for fname in os.listdir(txt_dir):
                fpath = os.path.join(txt_dir, fname)
                fi.write('{}\n'.format(fpath))
                num_files_to_tokenize += 1

        command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
                   '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
                   'json', '-outputDirectory', tokenized_data_dir]

        print("Tokenizing %i files in %s and saving in %s..." % (num_files_to_tokenize, txt_dir, tokenized_data_dir))
        subprocess.call(command)
        print("Stanford CoreNLP Tokenizer has finished.")
        os.remove("mapping_for_corenlp.txt")

        # Check that the tokenized data directory contains the same number of files as the original directory
        num_orig = len(os.listdir(txt_dir))
        num_tokenized = len(os.listdir(tokenized_data_dir))
        if num_orig != num_tokenized:
            raise Exception(
                "The tokenized data directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                    tokenized_data_dir, num_tokenized, root_data_dir, num_orig))
        print("Successfully finished tokenizing %s to %s.\n" % (root_data_dir, tokenized_data_dir))
        shutil.rmtree(txt_dir)

def tokenize(args):
    stories_dir = os.path.abspath(args.raw_path)
    tokenized_stories_dir = os.path.abspath(args.save_path)

    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if (not s.endswith('story')):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tokenized_stories_dir]
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)
    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    #print(abstract)
    #print(sents)
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge
    #print(max_rouge)
    return sorted(selected)

def cal_rouge(doc_sent_list, abstract_sent_list):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    doc = sum(doc_sent_list, [])
    doc = _rouge_clean(' '.join(doc)).split()
    #print(abstract)
    #print(sents)
    evaluated_1grams = _get_word_ngrams(1, [doc])
    reference_1grams = _get_word_ngrams(1, [abstract])
    #evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    #reference_2grams = _get_word_ngrams(2, [abstract])
    rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']

    #print(max_rouge)
    return rouge_1

def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


class RoBertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        #BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.sep_token = '<s>'
        self.cls_token = '</s>'
        self.pad_token = '<pad>'
        self.tgt_bos = '<s>'
        self.tgt_eos = '</s>'
        self.tgt_sent_split = '<s>'
        self.sep_vid = self.tokenizer.convert_tokens_to_ids(self.sep_token)
        self.cls_vid = self.tokenizer.convert_tokens_to_ids(self.cls_token)
        self.pad_vid = self.tokenizer.convert_tokens_to_ids(self.pad_token)
        self.tgt_bos_vid = self.tokenizer.convert_tokens_to_ids(self.tgt_bos)
        self.tgt_eos_vid = self.tokenizer.convert_tokens_to_ids(self.tgt_eos)
        self.tgt_sent_split_vid = self.tokenizer.convert_tokens_to_ids(self.tgt_sent_split)

    def preprocess(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False):

        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        #_segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        #segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        #segments_ids = []
        #for i, s in enumerate(segs):
        #    if (i % 2 == 0):
        #        segments_ids += s * [0]
        #    else:
        #        segments_ids += s * [1]
        segments_ids = [0]*len(src_subtoken_idxs)
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = '<s>' + ' </s>'.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt))) for tt in tgt]) + ' </s>'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt

class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False):

        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        #segments_ids = [0]*len(src_subtoken_idxs)
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt
             in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt

class PubmedData():
    def __init__(self, args):
        self.args = args
        model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False):

        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text,truncation=False)
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        assert len(segments_ids) == len(src_subtoken_idxs)
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_idxs = [i for i, s in enumerate(tgt) if (len(s) > self.args.min_src_ntokens_per_sent)]
        tg = [tgt[i][:self.args.max_src_ntokens_per_sent] for i in tgt_idxs]
        tgt_txt = [' '.join(tt) for tt in tg]

        tgt_text = ' {} {} '.format(self.sep_token, self.cls_token).join(tgt_txt)
        tgt_subtokens = self.tokenizer.tokenize(tgt_text, truncation=True)
        tgt_subtokens = [self.cls_token] + tgt_subtokens + [self.sep_token]
        if ((not is_test) and len(tgt_subtokens) < self.args.min_tgt_ntokens):
            return None
        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtokens)
        _segs = [-1] + [i for i, t in enumerate(tgt_subtoken_idxs) if t == self.sep_vid]
        tgt_segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        tgt_segments_ids = []
        for i, s in enumerate(tgt_segs):
            if (i % 2 == 0):
                tgt_segments_ids += s * [0]
            else:
                tgt_segments_ids += s * [1]
        assert len(tgt_segments_ids) == len(tgt_subtoken_idxs)
        tgt_cls_ids = [i for i, t in enumerate(tgt_subtoken_idxs) if t == self.cls_vid]

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt, tgt_segments_ids, tgt_cls_ids

class BioBertData():
    def __init__(self, args):
        self.args = args
        model_name = 'dmis-lab/biobert-v1.1'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False):

        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        #segments_ids = [0]*len(src_subtoken_idxs)
        assert len(segments_ids)==len(src_subtoken_idxs)
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt))) for tt
             in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt

def format_to_robert(args):
    print('... (5) Converting data to roBERT data... this will take a while')

    datasets = ['train', 'valid', 'test']

    #corpora = [os.path.join(args.raw_path, f) for f in os.listdir(args.raw_path)
    #           if not f.startswith('.') and f.endswith('.json')]
    print("")
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            #print("json_f:", json_f, real_name)
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        
        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_robert, a_lst):
            pass
        pool.close()
        pool.join()

def format_to_bert(args):
    print('... (5) Converting data to BERT data... this will take a while')

    datasets = ['train', 'valid', 'test']

    # corpora = [os.path.join(args.raw_path, f) for f in os.listdir(args.raw_path)
    #           if not f.startswith('.') and f.endswith('.json')]
    print("")
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            # print("json_f:", json_f, real_name)
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))

        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert, a_lst):
            pass
        pool.close()
        pool.join()

def format_to_pubmed_bert(args):
    print('... (5) Converting data to pubmed BERT data... this will take a while')

    datasets = ['train', 'valid', 'test']

    # corpora = [os.path.join(args.raw_path, f) for f in os.listdir(args.raw_path)
    #           if not f.startswith('.') and f.endswith('.json')]
    print("")
    for corpus_type in datasets:
        file = open(os.path.join(args.raw_path, '{}_feature'.format(corpus_type)), 'rb')
        data_feature = pickle.load(file)
        file.close()
        #print(type(data_feature))
        data_feature = data_feature.todense()
        a_lst = []
        i = 0
        print(data_feature.shape[0], data_feature.shape[1])
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            # print("json_f:", json_f, real_name)
            len_data = len(json.load(open(json_f)))
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt')),data_feature[i:i+len_data]))
            #print(i, len(data_feature[i:i+len_data]))
            i += len_data
        pool = Pool(args.n_cpus)
        for data in pool.imap(_format_to_pubmed_bert, a_lst):
            pass
        pool.close()
        pool.join()

def format_to_rouge(args):

    datasets = ['train', 'valid', 'test']

    # corpora = [os.path.join(args.raw_path, f) for f in os.listdir(args.raw_path)
    #           if not f.startswith('.') and f.endswith('.json')]
    f = open(join(args.raw_path, 'vocab.txt'), 'rb')
    vocab = []
    for line in f:
        vocab.append(line.strip())
    score = []
    for corpus_type in datasets:
        file = open(os.path.join(args.raw_path, '{}_feature'.format(corpus_type)), 'rb')
        data_feature = pickle.load(file)
        file.close()
        #print(type(data_feature))
        data_feature = data_feature.todense()
        a_lst = []
        i = 0
        print(data_feature.shape[0], data_feature.shape[1])
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            len_data = len(json.load(open(json_f)))
            jobs = json.load(open(json_file))
            for i,d in enumerate(jobs):
                each_feature = np.array(data_feature[i])
                index_v = np.argpartition(each_feature, -50)[-50:]
                top_word = []
                for idx in index_v:
                    top_word.append(vocab(top_word))
                each_score = cal_rouge(top_word, d['tgt'])
                score.append(each_score)
    print(np.array(score).mean())

def format_to_bio_bert(args):
    print('... (5) Converting data to BioBERT data... this will take a while')

    datasets = ['train', 'valid', 'test']

    # corpora = [os.path.join(args.raw_path, f) for f in os.listdir(args.raw_path)
    #           if not f.startswith('.') and f.endswith('.json')]
    print("")
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            # print("json_f:", json_f, real_name)
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))

        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bio_bert, a_lst):
            pass
        pool.close()
        pool.join()

def _format_to_robert(params):
    corpus_type, json_file, args, save_file = params
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    Robert = RoBertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']
        sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
        b_data = Robert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                 is_test=is_test)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)
        if (b_data is None):
            continue
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    gc.collect()

def _format_to_bert(params):
    corpus_type, json_file, args, save_file= params
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt= d['src'], d['tgt']
        if args.corpus != "pubmed":
            sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 6)
            #print(sent_labels)
            #break
        else:
            #sent_labels = label
            #if sent_labels == []:
            sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 7)
        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
        b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                 is_test=is_test)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

        if (b_data is None):
            continue
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    gc.collect()

def _format_to_pubmed_bert(params):
    corpus_type, json_file, args, save_file, data_feature = params
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    pubmed_bert = PubmedData(args)
    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for i,d in enumerate(jobs):
        source, tgt = d['src'], d['tgt']
        #print(source[:10], tgt[:5])
        sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 6)
        #print('labels:', sent_labels)
        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
        b_data = pubmed_bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,is_test=is_test)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

        if (b_data is None):
            continue
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt,tgt_segments_ids, tgt_cls_ids = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt,'tgt_segs':tgt_segments_ids,'tgt_cls_ids':tgt_cls_ids, 'data_feature': data_feature[i]}
        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    gc.collect()

def _format_to_bio_bert(params):
    corpus_type, json_file, args, save_file = params
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bio_bert = BioBertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']
        sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
        b_data = bio_bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                 is_test=is_test)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

        if (b_data is None):
            continue
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    gc.collect()

def format_to_lines(args):
    if args.corpus != 'pubmed':
        corpora = sorted([os.path.join(args.raw_path, f) for f in os.listdir(args.raw_path)
                      if not f.startswith('.') and not f.endswith('.abs.txt.json')])
        #train_files, valid_files, test_files = [], [], []:
        args_list = []
        for f_main in corpora:
            f_abs_name = '{}.abs.txt.json'.format(os.path.basename(f_main).split('.')[0])
            f_abs = os.path.join(args.raw_path, f_abs_name)
            args_list.append((f_main, f_abs, args))
        index_list = list(range(len(args_list)))
        random.shuffle(index_list)
        train_list_id = index_list[:int(len(args_list)*0.75)]
        eval_list_id = index_list[int(len(args_list)*0.75)+1:int(len(args_list)*0.9)]
        test_list_id = index_list[int(len(args_list)*0.9)+1:]
        train_files = [args_list[i] for i in train_list_id]
        valid_files = [args_list[i] for i in eval_list_id]
        test_files = [args_list[i] for i in test_list_id]
    else:
        root_data_dir = os.path.abspath(args.raw_path)
        train_files, valid_files, test_files = [], [], []
        test_txt_path = os.path.join(root_data_dir, 'test')
        val_txt_path = os.path.join(root_data_dir, 'val')
        train_txt_path = os.path.join(root_data_dir, 'train')
        test_corpora = sorted([os.path.join(test_txt_path, f) for f in os.listdir(test_txt_path)
                              if not f.startswith('.') and not f.endswith('.abs.txt.json')])
        val_corpora = sorted([os.path.join(val_txt_path, f) for f in os.listdir(val_txt_path)
                               if not f.startswith('.') and not f.endswith('.abs.txt.json')])
        train_corpora = sorted([os.path.join(train_txt_path, f) for f in os.listdir(train_txt_path)
                              if not f.startswith('.') and not f.endswith('.abs.txt.json')])
        for f_main in test_corpora:
            f_abs_name = '{}.abs.txt.json'.format(os.path.basename(f_main).split('.')[0])
            f_abs = os.path.join(test_txt_path, f_abs_name)
            test_files.append((f_main, f_abs, args))
        for f_main in val_corpora:
            f_abs_name = '{}.abs.txt.json'.format(os.path.basename(f_main).split('.')[0])
            f_abs = os.path.join(val_txt_path, f_abs_name)
            valid_files.append((f_main, f_abs, args))
        for f_main in train_corpora:
            f_abs_name = '{}.abs.txt.json'.format(os.path.basename(f_main).split('.')[0])
            f_abs = os.path.join(train_txt_path, f_abs_name)
            train_files.append((f_main, f_abs, args))

    start = time.time()
    print('... (4) Packing tokenized data into shards...')
    #print('Converting files count: {}'.format(len(corpora)))

    # imap executes in sync multiprocess manner
    # use array and shard_size to save the flow of ordered data
    corporas = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in ['train', 'valid', 'test']:
        a_lst = corporas[corpus_type]
        pool = Pool(args.n_cpus)
        dataset = []
        shard_count = 0
        with tqdm(total=len(a_lst)) as pbar:
            with tqdm(total=args.shard_size) as spbar:
                for i, data in enumerate(pool.imap(_format_to_lines, a_lst)):
                    if data:
                        dataset.append(data)
                    spbar.update()
                    if (len(dataset) > args.shard_size):
                        fpath = "{:s}/{:s}.{:d}.json".format(args.save_path, corpus_type, shard_count)
                        with open(fpath, 'w') as f:
                            f.write(json.dumps(dataset))
                        dataset = []
                        shard_count += 1
                        pbar.update()
                        spbar.reset()
                    
                 # gc.collect()
                spbar.close()
            pbar.close()
        pool.close()
        pool.join()
        if len(dataset) > 0:
            fpath = "{:s}/{:s}.{:d}.json".format(args.save_path, corpus_type, shard_count)
            print('last shard {} saved'.format(shard_count))
            with open(fpath, 'w') as f:
                f.write(json.dumps(dataset))
            shard_count += 1
    end = time.time()
    print('... Ending (4), time elapsed {}'.format(end - start))

def _format_to_lines(params):
    f_main, f_abs, args = params
    source, tgt= load_json(f_main, f_abs)
    if not source:
        return None
    else:
        return {'src': source, 'tgt': tgt}

def format_xsum_to_lines(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'test', 'valid']

    corpus_mapping = json.load(open(pjoin(args.raw_path, 'XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json')))

    for corpus_type in datasets:
        mapped_fnames = corpus_mapping[corpus_type]
        root_src = pjoin(args.raw_path, 'restbody')
        root_tgt = pjoin(args.raw_path, 'firstsentence')
        # realnames = [fname.split('.')[0] for fname in os.listdir(root_src)]
        realnames = mapped_fnames

        a_lst = [(root_src, root_tgt, n) for n in realnames]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_xsum_to_lines, a_lst):
            if (d is None):
                continue
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_xsum_to_lines(params):
    src_path, root_tgt, name = params
    f_src = pjoin(src_path, name + '.restbody')
    f_tgt = pjoin(root_tgt, name + '.fs')
    if (os.path.exists(f_src) and os.path.exists(f_tgt)):
        print(name)
        source = []
        for sent in open(f_src):
            source.append(sent.split())
        tgt = []
        for sent in open(f_tgt):
            tgt.append(sent.split())
        return {'src': source, 'tgt': tgt}
    return None
