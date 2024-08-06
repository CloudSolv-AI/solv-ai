import numpy as np
import unittest
from solv_ai import PrunedLayer

class TestPrunedLayer(unittest.TestCase):
    def test_forward(self):
        weights = np.array([[1, 2], [3, 4]])
        layer = PrunedLayer(weights, pruning_rate=0.5)
        x = np.array([[1, 1]])
        output = layer.forward(x)
        self.assertEqual(output.shape, (1, 2))

if __name__ == '__main__':
    unittest.main()