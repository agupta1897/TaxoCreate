"""
Ideally, you do NOT need to change any code in this file. 

After you implement all functions in "./distribution_methods/model.py", you generate evaluate results by running:
    python eval_DIH.py
"""

import argparse
import numpy as np
from tqdm import tqdm
from distribution_methods.model import DIHModel
from sklearn.metrics import average_precision_score


def read_vocab(file_path):
    entityID2entityPreferredName = {}
    entityID2AllNames = {}
    with open(file_path, "r") as fin:    
        for lid, line in enumerate(fin):
            if lid == 0:  # skip header
                continue
            line = line.strip()
            if line:
                segs = line.split("\t")
                entityID = segs[0]
                entityPreferredName = segs[1]
                entityAllNames = [" ".join(cname.split("_")) for cname in segs[2].split("||")]
                entityID2entityPreferredName[entityID] = entityPreferredName
                entityID2AllNames[entityID] = entityAllNames
    return entityID2entityPreferredName, entityID2AllNames


def read_testing_pairs(file_name, entityName2ID):
    test_pairs = []
    with open(file_name, "r") as fin:
        for line in fin:
            line = line.strip()
            if line:
                segs = line.split("\t")
                test_pairs.append([entityName2ID[segs[0]], entityName2ID[segs[1]], int(segs[2])])
    return test_pairs


def ap_at_k(y_true, y_score, k):
    """
    Computes AP@k, or AP of the model's top K predictions. Used in
    Shwartz, Santus and Schlectweg, EACL 2017.
    https://arxiv.org/abs/1612.04460
    """
    argsort = np.argsort(y_score)
    score_srt = y_score[argsort[-k:]]
    label_srt = y_true[argsort[-k:]]
    return average_precision_score(label_srt, score_srt)


def main(args):
    entityID2entityPreferredName, entityID2AllNames = read_vocab(args.vocab)
    entityName2ID = {}
    for k,v in entityID2AllNames.items():
        for t in v:
            t = "_".join(t.split(" "))
            entityName2ID[t] = k
    test_pairs = read_testing_pairs(args.test_data, entityName2ID)
    model = DIHModel(space_filename=args.feature_data)
    for metric_name in ["weeds_prec", "clarkeDE", "cosine", "invCL"]:
        print(f"Metric name: {metric_name}")
        trues = []
        preds = []
        for test_pair in tqdm(test_pairs):
            hypo = test_pair[1]
            hyper = test_pair[0]
            score = model.predict(hypo, hyper, metric_name)
            preds.append(score)
            trues.append(test_pair[-1])
        preds = np.array(preds)
        trues = np.array(trues)
        print(f"AP: {average_precision_score(trues, preds)}")
        print(f"AP@100: {ap_at_k(trues, preds, 100)}")
        print("="*80)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hypernym Extraction by Distributional Methods')
    parser.add_argument('-vocab', default="../raw_data/vocab.txt")
    parser.add_argument('-test_data', default="./test_pairs.txt")
    parser.add_argument('-feature_data', default="./entityID2contexts.txt")
    args = parser.parse_args()
    main(args)
