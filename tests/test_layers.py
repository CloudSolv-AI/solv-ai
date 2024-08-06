import numpy as np
import unittest
from solv_ai import DenseLayer

class TestDenseLayer(unittest.TestCase):
    def test_forward(self):
        dense = DenseLayer(10, 5)
        x = np.random.rand(3, 10)
        output = dense.forward(x)
        self.assertEqual(output.shape, (3, 5))

if __name__ == '__main__':
    unittest.main()