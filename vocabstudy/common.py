import csv
import sys
import os

HOME = '/home/lv71161/hlibbabii'
PATH_TO_DATASETS = os.path.join(HOME, 'raw_datasets')
PATH_TO_PREP_DATASETS = os.path.join(HOME, 'prep-datasets')
PATH_TO_PREP_DATASETS_TMP = '/tmp/scratch/prep-datasets'
VOCAB_STUDY_STATS_FILE = os.path.join(HOME, 'vocab-study-stats.csv')

sys.path.append(os.path.join(HOME, 'projects/log-recommender-dataprep/'))

import dataprep.api.text as text_api
from metrics.vector import get_new_vocab
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, Union, Any, List, Dict

from dataprep.api.corpus import PreprocessedCorpus
import logging
import regex
import pandas as pd
import shutil
import logging

logger = logging.getLogger(__name__)

IDENTIFIER_REGEX = "([[:lower:]]|[[:upper:]]|$|_)([[:lower:]]|[[:upper:]]|[0-9]|_|$)*"


def is_identifier(word: str, keywords: List[str]) -> bool:
    if word not in keywords and regex.fullmatch(IDENTIFIER_REGEX, word):
        return True
    else:
        return False


def without_non_identifiers(dct: Dict[str, int], keywords: List[str]) -> Dict[str, int]:
    res = {}
    for k, v in dct.items():
        if is_identifier(k, keywords):
            res[k] = v
    return res


# logging.disable(logging.WARNING)

@dataclass(frozen=True)
class Corpus(object):
    path: str
    extensions: str  # in format "py" or "java|c|py"


PrepCallable = Callable[..., PreprocessedCorpus]
ParametrizedPrepCallable = Callable[[Corpus], PreprocessedCorpus]


@dataclass(frozen=True)
class PrepFunction(object):
    callable: PrepCallable
    params: List = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)

    @property
    def apply(self) -> ParametrizedPrepCallable:
        def prep_corpus(corpus: Corpus, **kwargs):
            return self.callable(corpus.path, *self.params, **self.options, **kwargs,
                                 calc_vocab=True, extensions=corpus.extensions)

        return prep_corpus


sample = '''void test_WordUeberraschungPrinter() {
    if (eps >= 0.345e+4) { // FIXME 10L
        printWord("     ...     Überraschung 0x12");
    }
}'''


HEADER = [
    'Configuration', 'Train vocab', '% baseline', 'Train corpus size', '% baseline',
    '% OOV', '% OOV @200K', '% OOV @100K', '% OOV @75K', '% OOV @50K', '% OOV @25K',
    '% 1001 -', '% 101 - 1000', '% 11 - 100', '% 2-10', '%1' ,'total', '% 1-10'
]


class VocabStatsCsvWriter(object):
    def __init__(self, path, header: List[str]):
        self.path = path
        open(self.path, 'w').close() # rewriting the file
        self.write_line(header)

    def write_line(self, lst: List[str]) -> None:
        with open(self.path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(lst)


def print_oov_stats(train_vocab, test_vocab):
    tops = [sys.maxsize, 200000, 100000, 75000, 50000, 25000]
    csv_row = []
    for top in tops:
        test_tokens_number = sum(test_vocab.values())
        oov_in_test = get_new_vocab(list(train_vocab.items())[:top], list(test_vocab.items()))
        oov_tokens_number = sum(map(lambda e: e[1], oov_in_test))
        print(
            f"OOV in test set (top {top}): {len(oov_in_test)} ({100.0 * len(oov_in_test) / len(test_vocab):.2f}%), oov tokens number: {oov_tokens_number} ({100.0 * oov_tokens_number / test_tokens_number:.2f}%)")
        csv_row.append(f'{100.0 * len(oov_in_test) / len(test_vocab):.2f}')
    return csv_row


def chunks(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def percentiles(vocab: Dict[str, int]) -> Tuple[List[float], List[int]]:
    N_BUCKETS = 100
    values_list = list(vocab.values())
    lsts = chunks(values_list, N_BUCKETS)
    avg = [pd.Series(sequence).mean() for sequence in lsts]
    lsts = chunks(values_list, N_BUCKETS)
    med = [sequence[len(sequence) // 2] for sequence in lsts]
    return avg, med


def which_bucket(what: int, buckets: List[int]):
    for i in range(len(buckets)):
        if (i + 1 == len(buckets) or what < buckets[i + 1]) and what >= buckets[i]:
            return i
    raise ValueError(f'Invalid value: {what}, buckets: {buckets}')


def frequencies(vocab: Dict[str, int], buckets: List[int]):
    counts = [0 for _ in buckets]
    for k, v in vocab.items():
        counts[which_bucket(v, buckets)] += 1
    return counts

from math import sqrt 

def calc_variation_coeff(vocab: Dict[str, int]) -> float:
    n = len(vocab)
    mean = sum(vocab.values()) / float(n)
    variation_koeff = sqrt(sum(list(map(lambda v: (v - mean) ** 2, vocab.values())))) / n / mean 
    return variation_koeff
    

def plot_percentiles(vocab: Dict[str, int]) -> None:
    import plotly.graph_objects as go

    avg, med = percentiles(vocab)
    print("Average:")
    print(avg)
    df = pd.DataFrame({'percentile': list(range(100)), 'n_tokens': avg})

    data = [go.Bar(x=df.percentile, y=df.n_tokens)]
    fig = go.Figure(data=data)
    fig.update_layout(yaxis_type="log")
    fig.show()

    print("Median:")
    print(med)
    df = pd.DataFrame({'percentile': list(range(100)), 'n_tokens': med})

    data = [go.Bar(x=df.percentile, y=df.n_tokens)]
    fig = go.Figure(data=data)
    fig.update_layout(yaxis_type="log")
    fig.show()


def get_corpus_size(vocab: Dict[str, int]) -> int:
    return sum(vocab.values())


def calc_and_display_stats(prep_function, description, datasets: Tuple[str, str],
                           keywords: List[str], extension: str) -> List[str]:
    csv_row = []

    print(f"{description}\n")
    csv_row.append(description)

    vocabs = []
    prep_corpora = []
    for dataset in datasets:
        dataset_path = os.path.join(PATH_TO_DATASETS, dataset)
        prep_corpus = prep_function.apply(Corpus(dataset_path, extension), output_path=PATH_TO_PREP_DATASETS_TMP)
        prep_corpora.append(prep_corpus)
        vocabs.append(prep_corpus.load_vocab())
    path_to_train_dataset = prep_corpora[0].path_to_prep_dataset
    path_to_test_dataset = prep_corpora[1].path_to_prep_dataset
    logger.debug(f'Checking if train dataset exists: {path_to_train_dataset}')
    if os.path.exists(path_to_train_dataset):
        logger.info(f'Train dataset is found and is going to be removed to save space: {path_to_train_dataset}. Removing ...')
        shutil.rmtree(prep_corpora[0].path_to_prep_dataset)
    if os.path.exists(path_to_test_dataset):
        logger.info(f'Moving test dataset from tmp storage: {path_to_test_dataset} -> {PATH_TO_PREP_DATASETS}')
        path_after_move = os.path.join(PATH_TO_PREP_DATASETS, os.path.basename(path_to_test_dataset))
        if not os.path.exists(path_after_move):
            shutil.move(path_to_test_dataset, PATH_TO_PREP_DATASETS)
        else:
            logger.warning(f'Not moving. Path already exists: {path_after_move}')

    train_vocab, test_vocab = tuple(vocabs)
    print("\n========================   Split example   =========================\n")

    text_callable = getattr(text_api, prep_function.callable.__name__)
    print(text_callable(sample, *prep_function.params, **prep_function.options, extension="java"))

    print("\n========================   Stats           =========================\n")

    train_vocab_size = len(train_vocab)
    print(f"Train vocab: {train_vocab_size}")
    csv_row.append(str(train_vocab_size))
    csv_row.append('')
    test_vocab_identifiers = without_non_identifiers(test_vocab, keywords)

    print(
        f"Test vocab : {len(test_vocab)}, identifiers: {len(test_vocab_identifiers)} ({len(test_vocab_identifiers) / len(test_vocab):.2f}%)\n")

    train_corpus_size = get_corpus_size(train_vocab)
    print(f"Train corpus size: {train_corpus_size}")
    csv_row.append(str(train_corpus_size))
    csv_row.append('')

    test_tokens_number_ids = get_corpus_size(test_vocab_identifiers)
    test_tokens_number = get_corpus_size(test_vocab)
    print(
        f"Test corpus size : {test_tokens_number}, identifiers: {test_tokens_number_ids} ({test_tokens_number_ids / test_tokens_number:.2f}%)\n")

    csv_oov_stats = print_oov_stats(train_vocab, test_vocab)
    csv_row.extend(csv_oov_stats)
    #print("\nFor identifiers: \n")
    #print_oov_stats(train_vocab, test_vocab_identifiers)

    # plot_percentiles(train_vocab)
    buckets = [1, 2, 11, 101, 1001]
    freqs = frequencies(train_vocab, buckets)
    csv_row.extend(list(map(lambda f: f'{100.0 * f / train_vocab_size:.2f}', freqs)))
    csv_row.append('')
    csv_row.append('')
    variation_coeff = calc_variation_coeff(train_vocab)
    csv_row.append(variation_coeff)
    print(f"Variation coeff: {variation_coeff}")
    print(freqs)
    return csv_row
