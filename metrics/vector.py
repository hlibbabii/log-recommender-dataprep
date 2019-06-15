from collections import defaultdict, Counter
from typing import List, Tuple, Dict

from dataprep.bpepkg.merge import MergeList
from scipy.stats.stats import pearsonr
from nostril import nonsense


def convert_to_vectors(merges1: MergeList, merges2: MergeList) -> Tuple[List[int], List[int]]:
    d = defaultdict(lambda:[0,0])
    for merge in merges1:
        d[merge.pair][0] = merge.freq
    for merge in merges2:
        d[merge.pair][1] = merge.freq
    return zip(*[(v[0], v[1]) for v in d.values()])


def pearson(merges1: MergeList, merges2: MergeList) -> float:
    vector1, vector2 = convert_to_vectors(merges1, merges2)
    p = pearsonr(vector1, vector2)
    return p[0]


def cooccurences(*merges_lists: MergeList) -> Counter:
    c = Counter()
    for merges in merges_lists:
        c.update(list(map(lambda m: m.pair, merges)))
    return c


def summarize_cooccurences(cooccurences: Counter) -> Dict[int, float]:
    total_merges_in_n_chunks = defaultdict(int)
    for k, n in cooccurences.items():
        total_merges_in_n_chunks[n] += n

    total_merges = sum(total_merges_in_n_chunks.values())

    fraction_of_merges_in_n_chunks = {}
    for n, v in total_merges_in_n_chunks.items():
        fraction_of_merges_in_n_chunks[n] = float(v) / total_merges

    return fraction_of_merges_in_n_chunks


def merge_similarity_rate(vocab1: List[Tuple[str, int]], vocab2: List[Tuple[str, int]]) \
        -> Tuple[float, float, float, float, float]:
    if not vocab1:
        raise ValueError("Lists cannot be empty!")
    if len(vocab1) != len(vocab2):
        raise ValueError("Lists must be of the same length")

    total = 0
    different = 0
    more_subwords = 0
    split_now = 0
    less_subwords = 0
    unsplit_now = 0
    for (word1, freq1), (word2, freq2) in zip(vocab1, vocab2):
        if freq1 != freq2:
            raise ValueError(f"Not matching dict entries: {word1}: {freq1} and {word2}: {freq2}")

        if word1 == word2:
            total += freq1
        else:
            subwords1 = word1.split(" ")
            subwords2 = word2.split(" ")
            if "".join(subwords1) == "".join(subwords2):
                different += freq1
                total += freq1
                if len(subwords2) > len(subwords1):
                    more_subwords += freq1
                    if len(subwords1) == 1:
                        split_now += freq1
                elif len(subwords2) < len(subwords1):
                    less_subwords += freq1
                    if len(subwords2) == 1:
                        unsplit_now += freq1
            else:
                raise ValueError(f"Not matching dict entries: {word1}: {freq1} and {word2}: {freq2}")

    return float(different) / total, \
           float(more_subwords) / total, \
           float(less_subwords) / total, \
           float(split_now) / total, \
           float(unsplit_now) / total


def vocab_similarity_rate(vocab1: List[Tuple[str, int]], vocab2: List[Tuple[str, int]]) -> float:
    set1 = set([word for word, _ in vocab1])
    set2 = set([word for word, _ in vocab2])
    n_entries_in_common = len(set1.intersection(set2))
    return n_entries_in_common / len(set2) if set2 else 1.0


def nonsense_rate(vocab: List[Tuple[str, int]]):
    n_nonsense = 0
    for word, _ in vocab:
        if len(word) >= 6 and nonsense(word):
            n_nonsense += 1
    return n_nonsense / len(vocab) if vocab else 1.0
