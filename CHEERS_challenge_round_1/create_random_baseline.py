from ast import literal_eval
from collections import defaultdict, Counter
import argparse
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Creates a run file based on a random classifier.")
parser.add_argument("--train-path", default="data_round_1/sentences_en_train.csv",
                    help="The path to the train file.", type=str)
parser.add_argument("--evaluation-path", default="data_round_1/sentences_en_val.csv",
                    help="The path to the validation/test file.", type=str)
parser.add_argument("--output-path", default="runs/en_val_baseline_random.csv",
                    help="The path to save the output run file.", type=str)



args = parser.parse_args()

## Reading files
sentences_train = pd.read_csv(args.train_path)
sentences_evaluation = pd.read_csv(args.evaluation_path, usecols=["doc_id", "sentence_id"])
sentences_train["sector_ids"] = sentences_train["sector_ids"].apply(literal_eval) # convert `sector_ids` from string to list


## Random baseline for `is_relevant`
is_relevant_counts = sentences_train["is_relevant"].value_counts().to_dict()
is_relevant_counts = Counter(is_relevant_counts)
is_relevant_counts_zero_prob = is_relevant_counts[0] / (
    is_relevant_counts[0] + is_relevant_counts[1]
)
is_relevant_counts_one_prob = is_relevant_counts[1] / (
    is_relevant_counts[0] + is_relevant_counts[1]
)
sentences_evaluation["is_relevant"] = np.random.choice(
    [0, 1],
    size=len(sentences_evaluation),
    p=[is_relevant_counts_zero_prob, is_relevant_counts_one_prob],
)


## Random baseline for `sector_ids`
sector_id_freqs = defaultdict(int)
for sids in sentences_train["sector_ids"].tolist():
    for sid in sids:
        sector_id_freqs[sid] += 1
sector_id_freqs = Counter(sector_id_freqs)
# Calculate class probabilities for `sector_ids`
all_counts = sum(sector_id_freqs.values())
sector_id_probs = {
    cls_id: sector_id_freqs[cls_id] / all_counts for cls_id in sector_id_freqs.keys()
}
# Generate random samples according to the training data probabilities
choices = list(sector_id_probs.keys())
probs = list(sector_id_probs.values())
num_samples = sentences_evaluation["is_relevant"].eq(1).sum()
sentences_evaluation["sector_id"] = -1
sentences_evaluation.loc[sentences_evaluation["is_relevant"].eq(1), "sector_id"] = np.random.choice(
    choices, size=num_samples, p=probs
)

dir_path = os.path.dirname(args.output_path)

if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    
sentences_evaluation.to_csv(
    args.output_path,
    index=None,
    columns=["doc_id", "sentence_id", "is_relevant", "sector_id"],
)

print ("Output run file is saved at %s" % args.output_path)

