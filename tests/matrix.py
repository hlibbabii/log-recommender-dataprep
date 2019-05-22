import unittest

from metrics.matrix import cooccurence_matrix
from metrics.merge import Merge


def test_cooccurrence_matrix(self):
    merges1 = [Merge(('a', 'b'), 1), Merge(('c', 'd'), 1), Merge(('e', 'f'), 1)]
    merges2 = [Merge(('a', 'b'), 1), Merge(('c', 'd'), 1), Merge(('x', 'z'), 1)]
    merges3 = [Merge(('a', 'b'), 1), Merge(('p', 't'), 1), Merge(('l', 'm'), 1)]

    actual = cooccurence_matrix([merges1, merges2, merges3], [merges1, merges2, merges3])

    expected = [
        [1.0, 0.67, 0.33],
        [0.67, 1.0, 0.33],
        [0.33, 0.33, 1.0]
    ]

    for i in range(3):
        for j in range(3):
            self.assertAlmostEqual(expected[i][j], actual[i][j], places=2)


def test_cooccurrence_matrix_diff_merges_amount(self):
    merges1 = [Merge(('a', 'b'), 1)]
    merges2 = [Merge(('e', 'f'), 1)]

    merges1_bigger = [Merge(('a', 'b'), 1), Merge(('c', 'd'), 1)]
    merges2_bigger = [Merge(('e', 'f'), 1), Merge(('g', 'h'), 1)]

    actual = cooccurence_matrix([merges1, merges2], [merges1_bigger, merges2_bigger])

    expected = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    for i in range(2):
        for j in range(2):
            self.assertAlmostEqual(expected[i][j], actual[i][j], places=2)


if __name__ == '__main__':
    unittest.main()