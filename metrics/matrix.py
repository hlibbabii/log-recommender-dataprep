import csv
import os
from typing import List, Optional, Tuple, Dict

from dataprep.split import bpe_encode

from metrics.vector import pearson, cooccurences, merge_similarity_rate
from metrics.merge import Merge
from joblib import Memory

# needed for caching cell outputs
memory = Memory(cachedir=os.path.join(os.environ['HOME'], 'jupyter-cache'), verbose=0)


@memory.cache
def output_matrix_to_csv(matrix: List[List], path_to_csv: str) -> None:
    with open(path_to_csv, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([str(i) for i in range(len(matrix[0]))])
        for row in matrix:
            writer.writerow(row)

      
@memory.cache
def pearson_matrix(merge_lists: List[List[Merge]], path_to_save: Optional[str] = None) -> List[List[float]]:
    res = []
    for merges1 in merge_lists:
        row = []
        for merges2 in merge_lists:
            row.append(pearson(merges1, merges2))
        res.append(row)

    if path_to_save:
        output_matrix_to_csv(res, path_to_save)
    return res


@memory.cache
def cooccurence_matrix(merges_list1: List[List[Merge]], merges_list2: List[List[Merge]], path_to_save: Optional[str] = None):
    res = []
    for merges1 in merges_list1:
        row = []
        for merges2 in merges_list2:
            cooccs = cooccurences(merges1, merges2)
            merges_total = min(len(merges1), len(merges2))
            n_merges_present_in_both_lists = sum([1 for c in cooccs.values() if c == 2])
            row.append(float(n_merges_present_in_both_lists) / merges_total)
        res.append(row)
    if path_to_save:
        output_matrix_to_csv(res, path_to_save)
    return res


@memory.cache
def merge_similarity_rate_matrix(vocab_list: List[Dict[str, int]],
                                 merges_list: List[Dict[Tuple[str, str], int]],
                                 path_to_save: Optional[str] = None):
    res = []
    for idx, vocab in enumerate(vocab_list):
        row = []
        own_encoded_vocab = bpe_encode.encode(vocab, merges_list[idx])
        for merges in merges_list:
            cross_encoded_vocab = bpe_encode.encode(vocab, merges)
            row.append(merge_similarity_rate(cross_encoded_vocab.items(), own_encoded_vocab.items()))
        res.append(row)
    if path_to_save:
        output_matrix_to_csv(res, path_to_save)
    return res