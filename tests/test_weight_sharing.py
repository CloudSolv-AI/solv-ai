import torch
import unittest
from solv_ai import SharedWeightsLayer

class TestWeightSharing(unittest.TestCase):
    def test_shared_weights_layer(self):
        shared_weights = torch.randn(10, 10)
        shared_layer = SharedWeightsLayer(shared_weights)

        x = torch.randn(3, 10)
        output = shared_layer(x)
        self.assertEqual(output.shape, (3, 10))

if __name__ == '__main__':
    unittest.main()