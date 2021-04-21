#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import os
from pathlib import Path
from ast import literal_eval
import itertools
from urllib.parse import urlparse
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
from torch import nn
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

device = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess(
    in_folder: str = "data_round_1",
    out_folder: str = "preprocessed_data/",
    mode: str = "joint",
    output_label: str = None,
    explode_sector_ids: bool = False,
    verbose: int = 1
    ):
    """
    Preprocesses the raw data downloaded as the 
    CHEERS_challenge_round_1 folder.
    
    Parameters
    ----------
    in_folder: Path or str
        Path to the input data folder.
    out_folder: Path or str
        Path to the output data folder
    mode: str
        How to preprocess the data. This also has impact on the normalization. Can be "joint", TODO: add more, or not?
    output_label: str
        Decides the filename suffix of the files that will be created. If None then output_label = mode .
    explode_sector_ids: bool
        Wether to explode the sector_ids column (create an own column for each of the entries in the list)
        If true then also a column "sample_weight" is added so these samples can be potentially downweighted
    verbose : int
        Decides level of verbosity
    """

    # load data
    if verbose > 0: print("Loading data...", end="")
    sents_train = pd.read_csv(in_folder+"/sentences_en_train.csv", converters={'sector_ids': literal_eval})
    sents_val = pd.read_csv(in_folder+"/sentences_en_val.csv", converters={'sector_ids': literal_eval})
    sents_test = pd.read_csv(in_folder+"/sentences_en_test.csv", converters={'sector_ids': literal_eval})
    docs_train = pd.read_csv(in_folder+"/documents_en_train.csv")
    docs_val = pd.read_csv(in_folder+"/documents_en_val.csv")
    docs_test = pd.read_csv(in_folder+"/documents_en_test.csv")
    if verbose > 0: print("done")

    if verbose > 0: print("Indexing...", end="")
    docs_train = docs_train.set_index("doc_id")
    docs_val = docs_val.set_index("doc_id")
    docs_test = docs_test.set_index("doc_id")
    sents_train["sentence_position"] = sents_train["sentence_id"].apply(lambda x: np.log(x + 1))
    sents_val["sentence_position"] = sents_val["sentence_id"].apply(lambda x: np.log(x + 1))
    sents_test["sentence_position"] = sents_test["sentence_id"].apply(lambda x: np.log(x + 1))
    sents_train = sents_train.set_index(["doc_id","sentence_id"])
    sents_val = sents_val.set_index(["doc_id","sentence_id"])
    sents_test = sents_test.set_index(["doc_id","sentence_id"])
    if verbose > 0: print("done")


    if verbose > 0: print("Nominal Features to indices...", end="")
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
    if verbose > 0: print("done")

    if verbose > 0: print("Extract textual features...", end="")
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
    if verbose > 0: print("done")
    
    # tokenization
    if verbose > 0: print("Bert Tokenizing textual data...", end="")
    sents_train["tokenized_sentence"] = sents_train["sentence_text"].apply(lambda x:\
                                tokenizer(x, max_length=512, truncation="longest_first")["input_ids"])
    sents_val["tokenized_sentence"] = sents_val["sentence_text"].apply(lambda x:\
                                tokenizer(x, max_length=512, truncation="longest_first")["input_ids"])
    sents_test["tokenized_sentence"] = sents_test["sentence_text"].apply(lambda x:\
                                tokenizer(x, max_length=512, truncation="longest_first")["input_ids"])
    sents_train.drop("sentence_text", axis="columns", inplace= True)
    sents_val.drop("sentence_text", axis="columns", inplace= True)
    sents_test.drop("sentence_text", axis="columns", inplace= True)
    if verbose > 0: print("done")
    
    # remove unnecessary features
    if verbose > 0: print("Removing irrelevant features...", end="")
    docs_train.drop("lang_code",axis="columns", inplace = True)
    docs_val.drop("lang_code",axis="columns", inplace = True)
    docs_test.drop("lang_code",axis="columns", inplace = True)
    docs_train.drop("doc_text",axis="columns", inplace = True)
    docs_val.drop("doc_text",axis="columns", inplace = True)
    docs_test.drop("doc_text",axis="columns", inplace = True)
    docs_train.drop("doc_url",axis="columns", inplace = True)
    docs_val.drop("doc_url",axis="columns", inplace = True)
    docs_test.drop("doc_url",axis="columns", inplace = True)
    if verbose > 0: print("done")

    if mode == "joint":
        # join the tables
        if verbose > 0: print("Join data...", end="")
        joint_train = sents_train.join(docs_train, on="doc_id")
        joint_val = sents_val.join(docs_val, on="doc_id")
        joint_test = sents_test.join(docs_test, on="doc_id")
        if verbose > 0: print("done")

        if verbose > 0: print("Adding sample weights based on n_sector_ids...", end="")
        #Add weighting column to be able to downweight this samples (since in the scoring function these samples
        #are also only 50%)
        #(one could even think of 1/len(x)**2 because first ofd these samples should only influence 50% so much
        #since we split the samples
        #  AND they are also going to be weighted only 50% officially)
        joint_train["sample_weight"] = joint_train["sector_ids"].apply(lambda x: 1/len(x) if len(x) != 0 else 0)
        joint_val["sample_weight"] = joint_val["sector_ids"].apply(lambda x: 1/len(x) if len(x) != 0 else 0)
        # no sector ids key for the test data
        # joint_test["sample_weight"] = joint_test["sector_ids"].apply(lambda x: 1/len(x) if len(x) != 0 else 0)
        print("done")

        # normalization
        if verbose > 0: print("Normalizing data...", end="")
        # - docs
        scaler = StandardScaler().fit(joint_train[["sentence_count","text_length", "sentence_length", "sentence_position"]])
    
        joint_train[["sentence_count", "text_length", "sentence_length", "sentence_position"]] = scaler.transform(joint_train[["sentence_count", "text_length", "sentence_length", "sentence_position"]])
        joint_val[["sentence_count", "text_length", "sentence_length", "sentence_position"]] = scaler.transform(joint_val[["sentence_count", "text_length", "sentence_length", "sentence_position"]])
        joint_test[["sentence_count", "text_length", "sentence_length", "sentence_position"]] = scaler.transform(joint_test[["sentence_count", "text_length", "sentence_length", "sentence_position"]])

        if verbose > 0: print("done")

        #Exploding of target column sector ids
        if explode_sector_ids:
            joint_train = joint_train.explode("sector_ids")
            joint_val = joint_val.explode("sector_ids")
            joint_test = join_text.explode("sector_ids")

    
        if verbose > 0: print("Outputting data...", end="")
        Path(out_folder).mkdir(exist_ok=True, parents=True)

        if output_label is None:
            output_label = mode

        joint_train.to_hdf(os.path.join(out_folder, f"train_{output_label}.h5"), key='s', mode="w")
        joint_val.to_hdf(os.path.join(out_folder, f"validation_{output_label}.h5"), key='s', mode="w")
        joint_test.to_hdf(os.path.join(out_folder, f"test_{output_label}.h5"), key='s', mode="w")
        if verbose > 0: print("done")


class RelevantDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        target_mode: str = "isrelevant",
        device: str = "cpu",
        dimensions: tuple = None,
        load_only_relevant: bool = False
    ):
        """Constructor Function
        Parameters
        ----------
        dataset : str
            Decides which dataset will be loaded. Can be either "train", "test" or "val".
        target_mode : str
            Decides which target is returned in the __getitem__ function.
            Can be either "isrelevant", "sentencetype" or "both".TODO:!!!!
        device : str
            Decides on which device the torch tensors will be returned.
        dimensions : tuple
            The dimensions to use for returning one hot encodings.
        load_only_relevant : bool
            If true the Dataset will only contain samples with the "relevant" target equal True.
        """ 

        if dataset == "train":
            joint_dataframe = pd.read_hdf("./preprocessed_data/train_joint.h5", key="s")
        if dataset == "val":
            joint_dataframe = pd.read_hdf("./preprocessed_data/validation_joint.h5", key="s")
            if not dimensions:
                raise TypeError("Dimensions attribute is required for dataset type \"validation\".")
        if dataset == "test":
            joint_dataframe = pd.read_hdf("./preprocessed_data/test_joint.h5", key="s")
            if not dimensions:
                raise TypeError("Dimensions attribute is required for dataset type \"test\".")
        if load_only_relevant:
            joint_dataframe = joint_dataframe[joint_dataframe["is_relevant"] == True]

          
        if target_mode == "isrelevant":
            self.X = joint_dataframe[["sentence_position",
                                      "sentence_length",
                                      "tokenized_sentence", 
                                      "project_name", 
                                      "country_code",
                                      "url",
                                      "text_length",
                                      "sentence_count"]].to_numpy()
            self.Y = joint_dataframe["is_relevant"].to_numpy()
            if dimensions is None:
                self.dimensions = ((1, (4, 
                                        len(set(self.X[:,3])), 
                                        len(set(self.X[:,4])), 
                                        len(set(self.X[:,5])))),
                                   1)
            else:
                self.dimensions = dimensions

        if target_mode == "sentencetype":
            self.X = joint_dataframe[joint_dataframe["is_relevant"] == 1][["sentence_position",
                                                                           "sentence_length",
                                                                           "tokenized_sentence",
                                                                           "project_name", 
                                                                           "country_code", 
                                                                           "url", 
                                                                           "text_length",
                                                                           "sentence_count"]].to_numpy()
            joint_dataframe.loc[joint_dataframe["sector_ids"].apply(len) == 0, "sector_ids"] = 11
            joint_dataframe["sector_ids"] = joint_dataframe["sector_ids"].apply(lambda x: x[0] if type(x) != int else x)
            self.Y = joint_dataframe[joint_dataframe["is_relevant"] == 1]["sector_ids"].to_numpy()
            if dimensions is None:
                self.dimensions = ((1, (4, 
                                        len(set(joint_dataframe.to_numpy()[:,5])), 
                                        len(set(joint_dataframe.to_numpy()[:,6])),
                                        len(set(joint_dataframe.to_numpy()[:,7]))
                                       )
                                   ),
                                   len(set(self.Y[:])))
            else:
                self.dimensions = dimensions
            
        self.device = device
        
    def __len__(self):
        return len(self.Y)

    
    def __getitem__(self, idx, x_one_hot = True, x_train_ready = True):
        
        """
        Note that x_train_ready implies x_one_hot
        """
        x_tmp = self.X[idx]
        metric_x = torch.tensor([x_tmp[0], x_tmp[1], x_tmp[6], x_tmp[7]], device=self.device)#numerical features
        sentence_x = torch.tensor(x_tmp[2], device=self.device, dtype=torch.long)#bert features
        sentence_x = torch.cat((sentence_x, 
                                torch.zeros(512 - sentence_x.shape[0],
                                            device=self.device, 
                                            dtype= torch.long)))
        
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
        
        y = torch.tensor(self.Y[idx], device=self.device)

        if x_train_ready or x_one_hot:
            project_name_x = nn.functional.one_hot(project_name_x, num_classes = self.dimensions[0][1][1])
            country_code_x = nn.functional.one_hot(country_code_x, num_classes = self.dimensions[0][1][2])
            url_x = nn.functional.one_hot(url_x, num_classes = self.dimensions[0][1][3])
        if x_train_ready:
            x_other = torch.cat((metric_x, project_name_x, country_code_x, url_x), dim=0)
            return (sentence_x, x_other), y
        
        return (sentence_x, (metric_x, project_name_x, country_code_x, url_x)), y




