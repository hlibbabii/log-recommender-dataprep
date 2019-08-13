import sys
import os

HOME = '/home/lv71161/hlibbabii'
PATH_TO_DATASETS = os.path.join(HOME, 'raw_datasets')

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


def print_oov_stats(train_vocab, test_vocab):
    tops = [sys.maxsize, 200000, 100000, 75000, 50000, 25000]
    for top in tops:
        test_tokens_number = sum(test_vocab.values())
        oov_in_test = get_new_vocab(list(train_vocab.items())[:top], list(test_vocab.items()))
        oov_tokens_number = sum(map(lambda e: e[1], oov_in_test))
        print(
            f"OOV in test set (top {top}): {len(oov_in_test)} ({100.0 * len(oov_in_test) / len(test_vocab):.2f}%), oov tokens number: {oov_tokens_number} ({100.0 * oov_tokens_number / test_tokens_number:.2f}%)")


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


def calc_and_display_stats(prep_function, description, datasets: Tuple[str, str], keywords: List[str], extension: str) -> None:
    print(f"{description}\n")

    vocabs = []
    prep_corpora = []
    for dataset in datasets:
        dataset_path = os.path.join(PATH_TO_DATASETS, dataset)
        prep_corpus = prep_function.apply(Corpus(dataset_path, extension), output_path=os.path.join(HOME, 'prep-datasets'))
        prep_corpora.append(prep_corpus)
        vocabs.append(prep_corpus.load_vocab())
    print(f'Removing prepped dataset at {prep_corpora[0].path_to_prep_dataset}')
    shutil.rmtree(prep_corpora[0].path_to_prep_dataset)

    train_vocab, test_vocab = tuple(vocabs)
    print("\n========================   Split example   =========================\n")

    text_callable = getattr(text_api, prep_function.callable.__name__)
    print(text_callable(sample, *prep_function.params, **prep_function.options, extension="java"))

    print("\n========================   Stats           =========================\n")

    print(f"Train vocab: {len(train_vocab)}")
    test_vocab_identifiers = without_non_identifiers(test_vocab, keywords)

    print(
        f"Test vocab : {len(test_vocab)}, identifiers: {len(test_vocab_identifiers)} ({len(test_vocab_identifiers) / len(test_vocab):.2f}%)\n")

    print(f"Train corpus size: {get_corpus_size(train_vocab)}")
    test_tokens_number_ids = get_corpus_size(test_vocab_identifiers)
    test_tokens_number = get_corpus_size(test_vocab)
    print(
        f"Test corpus size : {test_tokens_number}, identifiers: {test_tokens_number_ids} ({test_tokens_number_ids / test_tokens_number:.2f}%)\n")

    print_oov_stats(train_vocab, test_vocab)
    print("\nFor identifiers: \n")
    print_oov_stats(train_vocab, test_vocab_identifiers)

    # plot_percentiles(train_vocab)
    buckets = [1, 2, 11, 101, 1001]
    freqs = frequencies(train_vocab, buckets)
    print(freqs)
