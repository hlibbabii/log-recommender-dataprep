from collections import defaultdict, Counter
from typing import List, Tuple

from scipy.stats.stats import pearsonr

from metrics.merge import Merge


def convert_to_vectors(merges1: List[Merge], merges2: List[Merge]) -> Tuple[List[int], List[int]]:
    d = defaultdict(lambda:[0,0])
    for merge in merges1:
        d[merge.pair][0] = merge.freq
    for merge in merges2:
        d[merge.pair][1] = merge.freq
    return zip(*[(v[0], v[1]) for v in d.values()])


def pearson(merges1: List[Merge], merges2: List[Merge]) -> float:
    vector1, vector2 = convert_to_vectors(merges1, merges2)
    p = pearsonr(vector1, vector2)
    return p[0]


def cooccurences(*merges_lists: List[Merge]) -> Counter:
    c = Counter()
    for merges in merges_lists:
        c.update(list(map(lambda m: m.pair, merges)))
    return c