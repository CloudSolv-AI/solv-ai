import torch
import torch.nn as nn
import unittest
from solv_ai import fuse_layers

class TestLayerFusion(unittest.TestCase):
    def test_fuse_layers(self):
        conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        bn = nn.BatchNorm2d(16)
        relu = nn.ReLU()
        fused_layer = fuse_layers(conv, bn, relu)

        x = torch.randn(1, 3, 224, 224)
        output = fused_layer(x)
        self.assertEqual(output.shape, (1, 16, 224, 224))

if __name__ == '__main__':
    unittest.main()