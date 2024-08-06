import torch
import unittest
from solv_ai import EfficientNet

class TestEfficientNet(unittest.TestCase):
    def test_forward(self):
        model = EfficientNet()
        x = torch.randn(3, 3, 224, 224)
        output = model(x)
        self.assertEqual(output.shape[0], 3)

if __name__ == '__main__':
    unittest.main()