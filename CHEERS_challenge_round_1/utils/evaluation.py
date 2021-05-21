
#Imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset

def make_prediction(
    relevant_model,
    sector_model,
    valid_ds,
    valid_path: str = "./preprocessed_data/validation_joint.h5",
    output_path: str = "predictions",
    batch_size: int = 64
    ):
    """
    Creates the prediction for the validation dataset.
    
    Parameters
    ----------
    relevant_model: Model
        Predicts the is_relevant feature
    sector_model: Model
        Predicts the sector_id features for samples with predicted
        is_relevant == 1
    valid_ds: Dataset
        Contains the validation data
    valid_path: Path or str
        Path to the preprocessed validation data
    output_path: Path or str
        Where to save the predicitons
    batch_size: Int
        Making the predictions
    """
    relevant_model.eval()
    sector_model.eval()
    device = next(relevant_model.parameters()).device
    sector_model.to(device)
    valid_ds.device = device
    df = pd.read_hdf(valid_path, key="s").reset_index()
    df = df[["doc_id", "sentence_id"]]
    df["is_relevant"] = [0 for _ in range(len(df))]
    df["sector_id"] = [-1 for _ in range(len(df))]
    valid_dl = DataLoader(valid_ds, batch_size=batch_size)
    pred = []
    for x, y in valid_dl:
        out = torch.sigmoid(relevant_model(x))
        pred.append((out > .5).detach())
    pred = torch.vstack(pred)
    idx = torch.where(pred)[0]
    print(len(idx))
    sector_ds = Subset(valid_ds, idx)
    sector_dl = DataLoader(sector_ds, batch_size=batch_size)
    pred = []
    for x, _ in sector_dl:
        out = sector_model(x)
        pred.append(out.detach())
    pred = torch.vstack(pred)
    pred = torch.argmax(pred, dim=-1)
    df.loc[idx, "is_relevant"] = 1
    df.loc[idx]["sector_id"] = list(pred)
    df.to_csv(output_path)
    return