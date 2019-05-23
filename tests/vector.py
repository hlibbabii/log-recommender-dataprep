import unittest

from metrics.merge import Merge
from metrics.vector import convert_to_vectors, merge_similarity_rate


class ConvertToVectorsTest(unittest.TestCase):
    def test_simple(self):
        merges1 = [Merge(('a', 'b'), 2), Merge(('c', 'd'), 3)]
        merges2 = [Merge(('a', 'b'), 7), Merge(('e', 'f'), 13)]

        actual = convert_to_vectors(merges1, merges2)

        self.assertEqual([(2, 3, 0),(7, 0, 13)], [a for a in actual])


class MergeSimilarityRateTest(unittest.TestCase):
    def test_simple(self):
        merge_list1 = [("ab", 3), ("cd", 3), ("ef", 4)]
        merge_list2 = [("ab", 3), ("cd", 3), ("e f", 4)]

        actual = merge_similarity_rate(merge_list1, merge_list2)
        expected = 0.6

        self.assertAlmostEqual(expected, actual, places=4)

    def test_different_entries(self):
        merge_list1 = [("ab", 3), ("cd", 3), ("ef", 4)]
        merge_list2 = [("different entry", 3), ("cd", 3), ("e f", 4)]

        with self.assertRaises(ValueError) as context:
            merge_similarity_rate(merge_list1, merge_list2)

    def test_inconsistent_entries(self):
        merge_list1 = [("ab", 3), ("cd", 3), ("ef", 4)]
        merge_list2 = [("a b", 30), ("cd", 3), ("e f", 4)]

        with self.assertRaises(ValueError) as context:
            merge_similarity_rate(merge_list1, merge_list2)

    def test_different_length(self):
        merge_list1 = [("ab", 3), ("cd", 3), ("ef", 4)]
        merge_list2 = [("a b", 3), ("cd", 3)]

        with self.assertRaises(ValueError) as context:
            merge_similarity_rate(merge_list1, merge_list2)


if __name__ == '__main__':
    unittest.main()