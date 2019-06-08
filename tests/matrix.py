import unittest

from dataprep.split.merge import Merge, MergeList

from metrics.matrix import cooccurence_matrix, merge_similarity_rate_matrix


class CooccurrenceMatrixTest(unittest.TestCase):
    def test_simple_pass(self):
        merges1 = MergeList().append(Merge(('a', 'b'), 1)).append(Merge(('c', 'd'), 1)).append(Merge(('e', 'f'), 1))
        merges2 = MergeList().append(Merge(('a', 'b'), 1)).append(Merge(('c', 'd'), 1)).append(Merge(('x', 'z'), 1))
        merges3 = MergeList().append(Merge(('a', 'b'), 1)).append(Merge(('p', 't'), 1)).append(Merge(('l', 'm'), 1))

        actual = cooccurence_matrix([merges1, merges2, merges3], [merges1, merges2, merges3])

        expected = [
            [1.0, 0.67, 0.33],
            [0.67, 1.0, 0.33],
            [0.33, 0.33, 1.0]
        ]

        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(expected[i][j], actual[i][j], places=2)

    def test_diff_merges_amount(self):
        merges1 = MergeList().append(Merge(('a', 'b'), 1))
        merges2 = MergeList().append(Merge(('e', 'f'), 1))

        merges1_bigger = MergeList().append(Merge(('a', 'b'), 1)).append(Merge(('c', 'd'), 1))
        merges2_bigger = MergeList().append(Merge(('e', 'f'), 1)).append(Merge(('g', 'h'), 1))

        actual = cooccurence_matrix([merges1, merges2], [merges1_bigger, merges2_bigger])

        expected = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]

        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(expected[i][j], actual[i][j], places=2)


class MergeSimilarityRateMatrixTest(unittest.TestCase):
    def test_trivial_pass(self):
        vocab_list = [{"ab": 10}]
        merges_list = [MergeList().append(Merge(('a', 'b'), 1))]

        actual = merge_similarity_rate_matrix(vocab_list, merges_list)
        expected = [[(0.0, 0.0, 0.0, 0.0, 0.0)]]

        self.assertEqual(expected, actual)

    def test_simple_pass(self):
        vocab_list = [{"abc": 10, "cd": 25, "gh": 5},
                      {"ef": 70, "ghu": 30}]
        merges_list = [MergeList().append(Merge(('a', 'b'), 1)), MergeList().append(Merge(('g', 'h'), 1)).append(Merge(('b', 'c'), 2))]

        actual = merge_similarity_rate_matrix(vocab_list, merges_list)
        expected = [[(0.0, 0.0, 0.0, 0.0, 0.0), (0.375, 0.0, 0.125, 0.0, 0.125)],
                    [(0.3, 0.3, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0)]]

        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()