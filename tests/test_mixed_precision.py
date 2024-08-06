import torch
import torch.nn as nn
import torch.optim as optim
import unittest
from solv_ai import MixedPrecisionLayer

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

class TestMixedPrecisionLayer(unittest.TestCase):
    def test_forward_and_backward(self):
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        mp_layer = MixedPrecisionLayer(model, optimizer)

        x = torch.randn(3, 10)
        output = mp_layer.forward(x)
        self.assertEqual(output.shape, (3, 5))

        loss = output.sum()
        mp_layer.backward(loss)

if __name__ == '__main__':
    unittest.main()