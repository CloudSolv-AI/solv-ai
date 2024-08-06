import torch
import torch.nn as nn
import unittest
from solv_ai import DynamicQuantizedLayer

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

class TestDynamicQuantization(unittest.TestCase):
    def test_forward(self):
        model = SimpleModel()
        dq_layer = DynamicQuantizedLayer(model)

        x = torch.randn(3, 10)
        output = dq_layer.forward(x)
        self.assertEqual(output.shape, (3, 5))

if __name__ == '__main__':
    unittest.main()