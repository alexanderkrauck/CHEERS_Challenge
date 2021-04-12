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
import torch
from torch.utils.data import Dataset
from torch import nn
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

    

class IsRelevantDataset(Dataset):
    def __init__(self, joint_dataframe: pd.DataFrame, device="cpu", dimensions = None):
        self.X = joint_dataframe[["sentence_position", "sentence_length", "tokenized_sentence", "project_name", "country_code", "url", "text_length", "sentence_count"]].to_numpy()
        self.Y = joint_dataframe["is_relevant"].to_numpy()
        self.device = device
        
        if dimensions is None:
            self.dimensions = ((1, (4, len(set(self.X[:,3])), len(set(self.X[:,4])), len(set(self.X[:,5])))), 2)
        else:
            self.dimensions = dimensions
        
    def __len__(self):
        return len(self.Y)

    
    def __getitem__(self, idx, x_one_hot = True, x_train_ready = True):
        
        """
        Note that x_train_ready implies x_one_hot
        """
        x_tmp = self.X[idx]
        metric_x = torch.tensor([x_tmp[0], x_tmp[1], x_tmp[6], x_tmp[7]], device=self.device)#numerical features
        sentence_x = torch.tensor(x_tmp[2], device=self.device, dtype=torch.long)#bert features
        sentence_x = torch.cat((sentence_x, torch.zeros(512 - sentence_x.shape[0], device=self.device, dtype= torch.long)))
        
        #one hot features:
        project_name_x = torch.tensor(x_tmp[3], device=self.device, dtype=torch.long)
        country_code_x = torch.tensor(x_tmp[4], device=self.device, dtype=torch.long)
        url_x = torch.tensor(x_tmp[5], device=self.device)
        
        y = torch.tensor(self.Y[idx], device=self.device, dtype=torch.long)

        if x_train_ready or x_one_hot:
            project_name_x = nn.functional.one_hot(project_name_x, num_classes = self.dimensions[0][1][1])
            country_code_x = nn.functional.one_hot(country_code_x, num_classes = self.dimensions[0][1][2])
            url_x = nn.functional.one_hot(url_x, num_classes = self.dimensions[0][1][3])
        if x_train_ready:
            x_other = torch.cat((metric_x, project_name_x, country_code_x, url_x), dim=0)
            return (sentence_x, x_other), y
        
        return (sentence_x, (metric_x, project_name_x, country_code_x, url_x)), y




