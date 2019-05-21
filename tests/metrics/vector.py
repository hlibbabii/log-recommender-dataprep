import unittest

from metrics.merge import Merge
from metrics.vector import convert_to_vectors


class VectorTest(unittest.TestCase):
    def test_convert_to_vectors(self):
        merges1 = [Merge(('a', 'b'), 2), Merge(('c', 'd'), 3)]
        merges2 = [Merge(('a', 'b'), 7), Merge(('e', 'f'), 13)]

        actual = convert_to_vectors(merges1, merges2)

        self.assertEqual([(2, 3, 0),(7, 0, 13)], [a for a in actual])


if __name__ == '__main__':
    unittest.main()