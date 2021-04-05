#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import os
from ast import literal_eval
import itertools
from urllib.parse import urlparse
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


class SentenceSet(Dataset):
    """Base dataset"""
    def __init__(self, sents_path, docs_path, preproc_func, ret_func, train):
        """
           *_path: path to the file
           preproc_func: the preprocessing function applied to the raw dataset
           ret_func: determines which features are returned
           train: true if training dataset
        """
        super(SentenceSet, self).__init__()
        self.ret_func = ret_func
        self.data = preproc_func(sents_path, docs_path, train)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.ret_func(self.data.iloc[idx])



def preprocess(sentences_path, documents_path, train=True):
    """preprocessor function"""
    
    # load data    
    sents = pd.read_csv(sentences_path, converters={'sector_ids': literal_eval})
    docs = pd.read_csv(documents_path)
    docs = docs.copy().set_index("doc_id")
    sents["position"] = sents["sentence_id"].apply(lambda x: np.log(x + 1))
    sents = sents.copy().copy().set_index(["doc_id","sentence_id"])
    
    # change nominal features to indices
    docs["doc_url"].fillna("",inplace=True)
    project_name_mapping = dict((o,idx) for idx, o in enumerate(set(docs["project_name"])))
    country_code_mapping = dict((o,idx) for idx, o in enumerate(set(docs["country_code"])))
    url_set = set(docs["doc_url"].apply(lambda x: urlparse(x).netloc))
    document_url_mapping = dict((o,idx) for idx, o in enumerate(url_set))
    docs.replace(project_name_mapping, inplace=True)
    docs.replace(country_code_mapping, inplace=True)
    docs["url"] = docs["doc_url"].apply(lambda x: urlparse(x).netloc).replace(document_url_mapping)
    if train == False:
        for item in docs.iterrows():
            if urlparse(item[1]["doc_url"]).netloc not in url_set:
                docs.loc[item[0], "url"] = len(url_set)
    
    
    # feature exctractor
    docs["text_lenght"] = docs["doc_text"].apply(len).apply(np.log)
    docs["sentence_count"] = sents.groupby(level="doc_id").size().apply(np.log)
    sents["sentence_lenght"] = sents["sentence_text"].apply(len).apply(np.log)
    
    # tokenization
    sents["tokenized_text"] = sents["sentence_text"].apply(lambda x: tokenizer(x, max_length=512, truncation="longest_first")["input_ids"])
    sents.drop("sentence_text", axis="columns", inplace= True)
    
    # remove unnecessary features
    docs.drop("lang_code",axis="columns", inplace = True)
    docs.drop("doc_text",axis="columns", inplace = True)
    docs.drop("doc_url",axis="columns", inplace = True)
    
    # join the tables
    joint = sents.join(docs, on="doc_id")
    
    return joint






