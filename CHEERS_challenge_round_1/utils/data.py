#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
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

device = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess():
    """preprocessor function"""
    # load data    
    DIR = "CHEERS_challenge_round_1"
    sents_train = pd.read_csv(DIR+"/sentences_en_train.csv", converters={'sector_ids': literal_eval})
    sents_val = pd.read_csv(DIR+"/sentences_en_val.csv", converters={'sector_ids': literal_eval})
    sents_test = pd.read_csv(DIR+"/sentences_en_test.csv", converters={'sector_ids': literal_eval})
    docs_train = pd.read_csv(DIR+"/documents_en_train.csv")
    docs_val = pd.read_csv(DIR+"/documents_en_val.csv")
    docs_test = pd.read_csv(DIR+"/documents_en_test.csv")
    docs_train = docs_train.copy().set_index("doc_id")
    docs_val = docs_val.copy().set_index("doc_id")
    docs_test = docs_test.copy().set_index("doc_id")
    sents_train["sentence_position"] = sents_train["sentence_id"].apply(lambda x: np.log(x + 1))
    sents_val["sentence_position"] = sents_val["sentence_id"].apply(lambda x: np.log(x + 1))
    sents_test["sentence_position"] = sents_test["sentence_id"].apply(lambda x: np.log(x + 1))
    sents_train = sents_train.copy().set_index(["doc_id","sentence_id"])
    sents_val = sents_val.copy().set_index(["doc_id","sentence_id"])
    sents_test = sents_test.copy().set_index(["doc_id","sentence_id"])
    
    # change nominal features to indices
    docs_train["doc_url"].fillna("",inplace=True)
    docs_val["doc_url"].fillna("",inplace=True)
    docs_test["doc_url"].fillna("",inplace=True)
    project_name_mapping = dict((o,idx) for idx, o in enumerate(set(docs_train["project_name"])))
    country_code_mapping = dict((o,idx) for idx, o in enumerate(set(docs_train["country_code"])))
    url_set = set(docs_train["doc_url"].apply(lambda x: urlparse(x).netloc))
    document_url_mapping = dict((o,idx) for idx, o in enumerate(url_set))
    docs_train.replace(project_name_mapping, inplace=True)
    docs_val.replace(project_name_mapping, inplace=True)
    docs_test.replace(project_name_mapping, inplace=True)
    docs_train.replace(country_code_mapping, inplace=True)
    docs_val.replace(country_code_mapping, inplace=True)
    docs_test.replace(country_code_mapping, inplace=True)
    docs_train["url"] = docs_train["doc_url"].apply(lambda x: urlparse(x).netloc).replace(document_url_mapping)
    docs_val["url"] = docs_val["doc_url"].apply(lambda x: urlparse(x).netloc).replace(document_url_mapping)
    docs_test["url"] = docs_test["doc_url"].apply(lambda x: urlparse(x).netloc).replace(document_url_mapping)
    for item in docs_val.iterrows():
        if urlparse(item[1]["doc_url"]).netloc not in url_set:
            docs_val.loc[item[0], "url"] = len(url_set)
    
    for item in docs_test.iterrows():
        if urlparse(item[1]["doc_url"]).netloc not in url_set:
            docs_test.loc[item[0], "url"] = len(url_set)
    
    # feature exctractor
    docs_train["text_length"] = docs_train["doc_text"].apply(len).apply(np.log)
    docs_val["text_length"] = docs_val["doc_text"].apply(len).apply(np.log)
    docs_test["text_length"] = docs_test["doc_text"].apply(len).apply(np.log)
    docs_train["sentence_count"] = sents_train.groupby(level="doc_id").size().apply(np.log)
    docs_val["sentence_count"] = sents_val.groupby(level="doc_id").size().apply(np.log)
    docs_test["sentence_count"] = sents_test.groupby(level="doc_id").size().apply(np.log)
    sents_train["sentence_length"] = sents_train["sentence_text"].apply(len).apply(np.log)
    sents_val["sentence_length"] = sents_val["sentence_text"].apply(len).apply(np.log)
    sents_test["sentence_length"] = sents_test["sentence_text"].apply(len).apply(np.log)
    
    t_l_mean = docs_train["text_length"].mean()
    t_l_std = docs_train["text_length"].std()
    s_c_mean = docs_train["sentence_count"].mean()
    s_c_std = docs_train["sentence_count"].std()
    s_l_mean = sents_train["sentence_length"].mean()
    s_l_std = sents_train["sentence_length"].std()
    s_p_mean = sents_train["sentence_position"].mean()
    s_p_std = sents_train["sentence_position"].std()
    
    docs_train["text_length"] = docs_train["text_length"].apply(lambda x: (x-t_l_mean)/t_l_std)
    docs_train["sentence_count"] = docs_train["sentence_count"].apply(lambda x: (x-s_c_mean)/s_c_std)
    sents_train["sentence_length"] = sents_train["sentence_length"].apply(lambda x: (x-s_l_mean)/s_l_std)
    sents_train["sentence_position"] = sents_train["sentence_position"].apply(lambda x: (x-s_p_mean)/s_p_std)
    
    
    # tokenization
    sents_train["tokenized_sentence"] = sents_train["sentence_text"].apply(lambda x:\
                                tokenizer(x, max_length=512, truncation="longest_first")["input_ids"])
    sents_val["tokenized_sentence"] = sents_val["sentence_text"].apply(lambda x:\
                                tokenizer(x, max_length=512, truncation="longest_first")["input_ids"])
    sents_test["tokenized_sentence"] = sents_test["sentence_text"].apply(lambda x:\
                                tokenizer(x, max_length=512, truncation="longest_first")["input_ids"])
    sents_train.drop("sentence_text", axis="columns", inplace= True)
    sents_val.drop("sentence_text", axis="columns", inplace= True)
    sents_test.drop("sentence_text", axis="columns", inplace= True)
    
    # remove unnecessary features
    docs_train.drop("lang_code",axis="columns", inplace = True)
    docs_val.drop("lang_code",axis="columns", inplace = True)
    docs_test.drop("lang_code",axis="columns", inplace = True)
    docs_train.drop("doc_text",axis="columns", inplace = True)
    docs_val.drop("doc_text",axis="columns", inplace = True)
    docs_test.drop("doc_text",axis="columns", inplace = True)
    docs_train.drop("doc_url",axis="columns", inplace = True)
    docs_val.drop("doc_url",axis="columns", inplace = True)
    docs_test.drop("doc_url",axis="columns", inplace = True)
    
    # join the tables
    joint_train = sents_train.join(docs_train, on="doc_id")
    joint_val = sents_val.join(docs_val, on="doc_id")
    joint_test = sents_test.join(docs_test, on="doc_id")
    
    if os.path.exists("preprocessed_data/"):
        shutil.rmtree("preprocessed_data/")
    os.mkdir("preprocessed_data/")
    
    joint_train.to_hdf(os.path.join("preprocessed_data", "train_joint.h5"), key='s')
    joint_val.to_hdf(os.path.join("preprocessed_data", "validation_joint.h5"), key='s')
    joint_test.to_hdf(os.path.join("preprocessed_data", "test_joint.h5"), key='s')


class RelevantDataset(Dataset):
    def __init__(self, device=device, dimensions = None, **flag):
    """flag: training: train=True
             validation: val=True
             test: test=True
             samples with is_relevant == True: only_relevant=True
     """
        if "train" in flag:
            joint_dataframe = pd.read_hdf("preprocessed_data/train_joint.h5", key="s")
        if "val" in flag:
            joint_dataframe = pd.read_hdf("preprocessed_data/val_joint.h5", key="s")
        if "test" in flag:
            joint_dataframe = pd.read_hdf("preprocessed_data/test_joint.h5", key="s")
        if "only_relevant" in flag:
            joint_dataframe = joint_dataframe[joint_dataframe["is_relevant"]==True]
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
        
      
      
preprocess()
    

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




