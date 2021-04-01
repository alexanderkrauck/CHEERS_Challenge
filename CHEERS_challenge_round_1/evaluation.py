import argparse
from ast import literal_eval

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


example_text = """example:

python eval.py --ground-truth-path data_round_1/sentences_en_val.csv --runfile-path runs/en_val_baseline_random.csv"""

parser = argparse.ArgumentParser(
    description="Calculate the evaluation metrics.",
    epilog=example_text,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "-gt", "--ground-truth-path", help="The ground truth csv file.", type=str, required=True
)
parser.add_argument(
    "-r", "--runfile-path", help="The prediction csv file.", type=str, required=True
)

args = parser.parse_args()

df_true = pd.read_csv(
    args.ground_truth_path, usecols=["doc_id", "sentence_id", "is_relevant", "sector_ids"],
)
df_pred = pd.read_csv(
    args.runfile_path, usecols=["doc_id", "sentence_id", "is_relevant", "sector_id"],
)


def hamming_score(y_true, y_pred):
    # https://link.springer.com/chapter/10.1007/978-3-540-24775-3_5
    score = 0
    empty_true = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == []:
            empty_true += 1
            continue
        yt = set(yt)
        score += 1 / (len(yt) + 1)
    return score / (len(y_true) - empty_true)

def evaluate (df_pred, df_true):
    # sanity check #1
    if len(df_pred) != len(df_true):
        print("Both files must have same number of instances")
        exit()

    # sanity check #2
    if (~df_pred[df_pred["is_relevant"].eq(0)]["sector_id"].eq(-1)).sum():
        print("when is_relevant is predicted as 0, sector_id must be -1")
        exit()

    df_true.loc[:, "sector_ids"] = df_true["sector_ids"].apply(literal_eval)
    possible_sector_ids = set()
    for sids in df_true["sector_ids"].tolist():
        possible_sector_ids.update(sids)

    # sanity check #3
    if (
        ~(df_pred[df_pred["is_relevant"].eq(1)]["sector_id"].isin(possible_sector_ids))
    ).sum():
        print("when is_relevant is predicted as 1, sector_id must be a valid sector_id")
        exit()


    df_eval = pd.merge(
        df_true,
        df_pred,
        on=["doc_id", "sentence_id"],
        how="left",
        suffixes=("_true", "_pred"),
    )
    y_true_relevance, y_pred_relevance = (
        df_eval["is_relevant_true"].tolist(),
        df_eval["is_relevant_pred"].tolist(),
    )
    # y_pred_relevance = list(map(lambda x: -1 if pd.isna(x) else x, y_pred_relevance))
    _, _, f1_score_relevance, _ = precision_recall_fscore_support(
        y_true_relevance, y_pred_relevance, average="macro", warn_for=tuple()
    )

    print(
        f"Macro-averaged F1 score for is_relevant variable is: {f1_score_relevance*100:.2f}"
    )

    y_true_sectorids = df_eval[df_eval["is_relevant_pred"].eq(1)]["sector_ids"].tolist()
    y_pred_sectorids = df_eval[df_eval["is_relevant_pred"].eq(1)]["sector_id"].tolist()
    # y_pred_sectorids = list(map(lambda x: [] if pd.isna(x) else x, y_pred_sectorids))
    if y_pred_sectorids:
        hamming_score_sectorids = hamming_score(y_true_sectorids, y_pred_sectorids)
    else:
        hamming_score_sectorids = 0
    print(f"Accuracy for sector_ids variable is: {hamming_score_sectorids*100:.2f}")

    hum_impact = 0.5 * f1_score_relevance + 0.5 * hamming_score_sectorids
    print(f"HumImpact Score is: {hum_impact*100:.2f}")
    
    return {'relevance_f1_score_macro': f1_score_relevance,
           'sectorids_accuracy': hamming_score_sectorids,
           'HumImpact': hum_impact}


print (evaluate (df_pred, df_true))