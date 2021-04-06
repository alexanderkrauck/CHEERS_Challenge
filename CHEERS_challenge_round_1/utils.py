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
    
    mappings = None
    
    def __init__(self, sents_path, docs_path, preproc_func, ret_func, train=True):
        """
           path: path to the file
           preproc_func: the preprocessing function applied to the raw dataset
           ret_func: determines which features are returned
        """
        super(SentenceSet, self).__init__()
        if train:
            self.extract_mappings(sents_path, docs_path)
        self.ret_func = ret_func
        self.data = preproc_func(sents_path, docs_path, self.mappings, train)
        
    @classmethod
    def extract_mappings(cls, sents_path, docs_path):
        cls.mappings = {}
        sents = pd.read_csv(sents_path)
        docs = pd.read_csv(docs_path)
        project_name_mapping = dict((o,idx) for idx, o in enumerate(set(docs["project_name"])))
        country_code_mapping = dict((o,idx) for idx, o in enumerate(set(docs["country_code"])))
        docs["doc_url"].fillna("",inplace=True)
        url_set = set(docs["doc_url"].apply(lambda x: urlparse(x).netloc))
        document_url_mapping = dict((o,idx) for idx, o in enumerate(url_set))
        cls.mappings["project_name_mapping"] = project_name_mapping
        cls.mappings["country_code_mapping"] = country_code_mapping
        cls.mappings["url_set"] = url_set
        cls.mappings["document_url_mapping"] = document_url_mapping
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.ret_func(self.data.iloc[idx])



def preprocess(sentences_path, documents_path, mappings, train=True):
    """preprocessor function"""
    if mappings == None:
        raise BaseException("first create training dataset")
    
    # load data    
    sents = pd.read_csv(sentences_path, converters={'sector_ids': literal_eval})
    docs = pd.read_csv(documents_path)
    docs = docs.copy().set_index("doc_id")
    sents["position"] = sents["sentence_id"].apply(lambda x: np.log(x + 1))
    sents = sents.copy().copy().set_index(["doc_id","sentence_id"])
    
    # change nominal features to indices
    docs["doc_url"].fillna("",inplace=True)
    docs.replace(mappings["project_name_mapping"], inplace=True)
    docs.replace(mappings["country_code_mapping"], inplace=True)
    docs["url"] = docs["doc_url"].apply(lambda x: urlparse(x).netloc).replace(mappings["document_url_mapping"])
    if train == False:
        for item in docs.iterrows():
            if urlparse(item[1]["doc_url"]).netloc not in mappings["url_set"]:
                docs.loc[item[0], "url"] = len(mappings["url_set"])
    
    
    # feature exctractor
    docs["text_lenght"] = docs["doc_text"].apply(len).apply(np.log)
    docs["sentence_count"] = sents.groupby(level="doc_id").size().apply(np.log)
    sents["sentence_lenght"] = sents["sentence_text"].apply(len).apply(np.log)
    
    # tokenization
    sents["tokenized_text"] = sents["sentence_text"].apply(lambda x:\
                                tokenizer(x, max_length=512, truncation="longest_first")["input_ids"])
    sents.drop("sentence_text", axis="columns", inplace= True)
    
    # remove unnecessary features
    docs.drop("lang_code",axis="columns", inplace = True)
    docs.drop("doc_text",axis="columns", inplace = True)
    docs.drop("doc_url",axis="columns", inplace = True)
    
    # join the tables
    joint = sents.join(docs, on="doc_id")
    
    return joint






