import pickle as pkl
import json
import re
import math
import os
import subprocess
import csv
import shutil
import numpy as np
import scipy.sparse as sp
from math import log
from collections import defaultdict
import pandas as pd
from os.path import join, exists
from tqdm import tqdm
#from stop_words import get_stop_words
from nltk.corpus import stopwords
from collections import defaultdict
import nltk
import argparse
import glob
import collections

parser = argparse.ArgumentParser()
parser.add_argument("-raw_path", default='./raw_data')
parser.add_argument("-save_path", default='./token_data/')
args = parser.parse_args()


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)
    return adj_normalized

def clean_data(docs_list):
    clean_text_path = join(args.raw_path, 'sentences_clean.txt')
    #if not exists(clean_text_path):
    word_counts = defaultdict(int)
    for doc in docs_list:
        join_doc = ' '.join(doc)
            #print(join_doc)
        temp = clean_doc(join_doc)
        words = temp.split()
        for word in words:
             word_counts[word] += 1
    clean_docs = clean_documents(docs_list, word_counts)
    corpus_str = '\n'.join(clean_docs)
    f = open(clean_text_path, 'w')
    f.write(corpus_str)
    f = open(clean_text_path, 'r')
    lines = f.readlines()
    min_len = 10000
    aver_len = 0
    max_len = 0
    for line in lines:
        line = line.strip()
        temp = line.split()
        #print('temp:',len(temp))
        aver_len += len(temp)
        if len(temp) < min_len:
            min_len = len(temp)
        if len(temp) > max_len:
            max_len = len(temp)
    f.close()
    print(len(lines))
    aver_len = 1.0 * aver_len / len(lines)
    print('min_len : ' + str(min_len))
    print('max_len : ' + str(max_len))
    print('average_len : ' + str(aver_len))
    return clean_docs


def clean_documents(docs, word_counts):
    #nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    #print(stop_words)
    ret = []
    for doc in docs:
        doc = ' '.join(doc)
        doc = clean_doc(doc)
        words = doc.split()
        words = [word for word in words if word not in stop_words and word_counts[word] >= 5]
        doc = ' '.join(words).strip()
        if doc != '':
            ret.append(' '.join(words).strip())
        else:
            ret.append(' ')
    return ret


def clean_doc(string):
    string = re.sub(r"^\"", "", string)
    string = re.sub(r"\"$", "", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\.", " ", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            if token in stopwords.words('english'):
                break
            token = token.strip().split()[0]
            vocab[token] = index
            index += 1
    return vocab

def get_vocab(text_list):
    word_freq = defaultdict(int)
    for doc_words in text_list:
        words = doc_words.split()
        for word in words:
           word_freq[word] += 1
    return word_freq

def build_word_doc_edges(doc_list,vocab):
    # build all docs that a word is contained in
    words_in_docs = defaultdict(set)
    for i, doc_words in enumerate(doc_list):
        words = doc_words.split()
        for word in words:
            if word in vocab:
                words_in_docs[word].add(i)
    print("words_in_docs:",len(words_in_docs.items()))
    word_doc_freq = {}
    for word, doc_list in words_in_docs.items():
        word_doc_freq[word] = len(doc_list)

    return words_in_docs, word_doc_freq

def build_graph(doc_list, word_id_map, vocab, window_size=20):
    # constructing all windows
    windows = []
    for doc_words in doc_list:
        words = doc_words.split()
        words = [filt_word for filt_word in words if filt_word in vocab]
        doc_length = len(words)
        #if words in vocab:
        if doc_length <= window_size:
            windows.append(words)
        else:
            for i in range(doc_length - window_size + 1):
                window = words[i: i + window_size]
                windows.append(window)
    # constructing all single word frequency
    word_window_freq = defaultdict(int)
    for window in windows:
        appeared = set()
        for word in window:
            if word not in appeared:
                word_window_freq[word] += 1
                appeared.add(word)
    #print('word window freq:',word_window_freq)
    # constructing word pair count frequency
    word_pair_count = defaultdict(int)
    for window in tqdm(windows):
        for i in range(1, len(window)):
            for j in range(i):
                word_i = window[i]
                word_j = window[j]
                word_i_id = word_id_map[word_i]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_count[(word_i_id, word_j_id)] += 1
                word_pair_count[(word_j_id, word_i_id)] += 1
    w_row = []
    w_col = []
    w_weight = []
    # pmi as weights
    num_window = len(windows)
    #print('word_pair_count:', word_pair_count.items())
    for word_id_pair, count in tqdm(word_pair_count.items()):
        i, j = word_id_pair[0], word_id_pair[1]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        print('pmi:', pmi)
        if pmi <= 0:
            continue
        w_row.append(i)
        w_col.append(j)
        w_weight.append(pmi)
    number_nodes = len(vocab)
    adj_mat = sp.csr_matrix((w_weight, (w_row, w_col)), shape=(number_nodes, number_nodes))
    adj = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat) - adj_mat.multiply(adj_mat.T > adj_mat)
    return adj

def build_feature(doc_list, word_id_map, vocab, word_doc_freq, train_len, val_len, test_len):
    # frequency of document word pair
    doc_word_freq = defaultdict(int)
    for i, doc_words in enumerate(doc_list):
        words = doc_words.split()
        for word in words:
            if word in vocab:
                word_id = word_id_map[word]
                doc_word_str = (i, word_id)
                doc_word_freq[doc_word_str] += 1

    train_row = []
    train_col = []
    train_weight = []
    val_row = []
    val_col = []
    val_weight = []
    test_row = []
    test_col = []
    test_weight = []

    num_docs = len(doc_list)

    for i, doc_words in enumerate(doc_list):
        words = doc_words.split()
        train_doc_word_set = set()
        val_doc_word_set = set()
        test_doc_word_set = set()
        for word in words:
            #if word in doc_word_set:
            #    continue
            if word not in vocab:
                continue
            word_id = word_id_map[word]
            freq = doc_word_freq[(i, word_id)]
           # print('freq:', freq)
            if i < train_len:
                if word in train_doc_word_set:
                    continue
                train_row.append(i)
                train_col.append(word_id)
                idf = log(1.0 * num_docs /
                      word_doc_freq[vocab[word_id]])
                #print('train_idf:', idf)
                train_weight.append(freq * idf)
                train_doc_word_set.add(word)
            if train_len<=i and i< train_len+val_len:
                if word in val_doc_word_set:
                    continue
                #print(i, word)
                val_row.append(i-train_len)
                val_col.append(word_id)
                idf = log(1.0 * num_docs /
                          word_doc_freq[vocab[word_id]])
                #print('val_idf:', idf)
                val_weight.append(freq * idf)
                val_doc_word_set.add(word)
            if train_len+val_len<=i:
                if word in test_doc_word_set:
                    continue
                #print(i, word)
                test_row.append(i-train_len-val_len)
                test_col.append(word_id)
                idf = log(1.0 * num_docs /
                          word_doc_freq[vocab[word_id]])
                #print('test_idf:', idf)
                test_weight.append(freq * idf)
                test_doc_word_set.add(word)

    train_feature = sp.csr_matrix((train_weight, (train_row, train_col)), shape=(train_len, len(vocab)))
    val_feature = sp.csr_matrix((val_weight, (val_row, val_col)), shape=(val_len, len(vocab)))
    test_feature = sp.csr_matrix((test_weight, (test_row, test_col)), shape=(test_len, len(vocab)))
    return train_feature, val_feature, test_feature


datasets = ['train', 'valid', 'test']
data_len  = []
src = []
for corpus_type in datasets:
    length = 0
    is_test = corpus_type == 'test'
    for json_file in glob.glob(args.raw_path + '/' + corpus_type + '.[0-9]*.json'):
        jobs = json.load(open(json_file))
        for data in jobs:
            source, tgt = data['src'], data['tgt']
            sr = [' '.join(s) for s in source]
            if not is_test:
                sr += [' '.join(s) for s in tgt]
            src.append(sr)
            length += 1
    data_len.append(length)

print(len(src), data_len[0], data_len[1],data_len[2])
#src = src[:200]
train_len = data_len[0]
val_len = data_len[1]
test_len = data_len[2]
all_src = clean_data(src)
print(len(all_src))
if not exists(join(args.raw_path, 'vocab.txt')):
    word_freq = get_vocab(all_src)
    sorted_x = sorted(word_freq.items(), key=lambda x:-x[1])[:60000]
    sorted_dict = collections.OrderedDict(sorted_x)
    word_freq = sorted_dict
    vocab = list(word_freq.keys())
    print(len(vocab))
    vocab_str = '\n'.join(vocab)
    f = open(join(args.raw_path, 'vocab.txt'), 'w')
    f.write(vocab_str)
    f.close()
else:
    f = open(join(args.raw_path, 'vocab.txt'), 'rb')
    vocab = []
    for line in f:
        vocab.append(line.strip())

#print('vocab:', vocab)
words_in_docs, word_doc_freq = build_word_doc_edges(all_src, vocab)
word_id_map = {word: i for i, word in enumerate(vocab)}

#if not exists(join(args.raw_path, 'word_graph')):
#    word_graph = build_graph(all_src, word_id_map, vocab)
#    f = open(os.path.join(args.raw_path,'word_graph'), 'wb')
#    pkl.dump(word_graph, f)
#    f.close()
#else:
#    f = open(os.path.join(args.raw_path,'word_graph'), 'rb')
#    word_graph = pkl.load(f)
#    f.close()
#print(word_graph)
if not exists(join(args.raw_path, 'train_feature')):
    train_feature, val_feature, test_feature \
    = build_feature(all_src, word_id_map, vocab, word_doc_freq, train_len, val_len, test_len)
else:
    f = open(os.path.join(args.raw_path, 'train_feature'), 'rb')
    train_feature = pkl.load(f)
    f.close()
    f = open(os.path.join(args.raw_path, 'valid_feature'), 'rb')
    val_feature = pkl.load(f)
    f.close()
    f = open(os.path.join(args.raw_path, 'test_feature'), 'rb')
    test_feature = pkl.load(f)
    f.close()
#print(train_feature.shape[0], train_feature.shape[1],val_feature.shape[0], val_feature.shape[1], test_feature.shape[0],test_feature.shape[1],word_graph.shape[0], word_graph.shape[1])
#normalize_graph = preprocess_adj(word_graph)
#print(normalize_graph)
train_feature = preprocess_features(train_feature)
print(train_feature)
#train_feature=train_feature.dot(normalize_graph)

#train_feature = train_feature.dot(normalize_graph)

val_feature = preprocess_features(val_feature)#.dot(normalize_graph)
#val_feature = val_feature.dot(normalize_graph)
test_feature= preprocess_features(test_feature)#.dot(normalize_graph)
#test_feature = test_feature.dot(normalize_graph)
print("train:", train_feature)
print("val:", val_feature)
print("test:", test_feature)
#print(word_graph)
#print(train_feature.shape[0], train_feature.shape[1],val_feature.shape[0], val_feature.shape[1], test_feature.shape[0],test_feature.shape[1],word_graph.shape[0], word_graph.shape[1])
f = open(os.path.join(args.raw_path,'train_feature'), 'wb')
pkl.dump(train_feature, f)
f.close()

f = open(os.path.join(args.raw_path,'valid_feature'), 'wb')
pkl.dump(val_feature, f)
f.close()

f = open(os.path.join(args.raw_path,'test_feature'), 'wb')
pkl.dump(test_feature, f)
f.close()
