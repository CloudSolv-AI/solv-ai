import numpy as np
import unittest
from solv_ai import QuantizedLayer, PrunedLayer, DenseLayer

class TestModelComponents(unittest.TestCase):
    def setUp(self):
        # Create a basic DenseLayer model for testing
        self.input_size = 10
        self.output_size = 5
        self.dense_layer = DenseLayer(self.input_size, self.output_size)

    def test_dense_layer(self):
        x = np.random.rand(3, self.input_size)
        output = self.dense_layer.forward(x)
        self.assertEqual(output.shape, (3, self.output_size))

    def test_quantized_layer(self):
        weights = np.random.rand(self.output_size, self.input_size)
        quant_layer = QuantizedLayer(weights, bits=8)
        x = np.random.rand(3, self.input_size)
        output = quant_layer.forward(x)
        self.assertEqual(output.shape, (3, self.output_size))

    def test_pruned_layer(self):
        weights = np.random.rand(self.output_size, self.input_size)
        prune_layer = PrunedLayer(weights, pruning_rate=0.5)
        x = np.random.rand(3, self.input_size)
        output = prune_layer.forward(x)
        self.assertEqual(output.shape, (3, self.output_size))

if __name__ == '__main__':
    unittest.main()