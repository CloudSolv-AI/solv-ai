import numpy as np
import unittest
from solv_ai import QuantizedLayer

class TestQuantizedLayer(unittest.TestCase):
    def test_forward(self):
        weights = np.array([[1, 2], [3, 4]])
        layer = QuantizedLayer(weights, bits=8)
        x = np.array([[1, 1]])
        output = layer.forward(x)
        self.assertEqual(output.shape, (1, 2))

if __name__ == '__main__':
    unittest.main()