import torch
import torch.nn as nn
import unittest
from solv_ai import fold_batch_norm

class TestBNFolding(unittest.TestCase):
    def test_fold_batch_norm(self):
        conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        bn = nn.BatchNorm2d(16)
        folded_conv = fold_batch_norm(conv, bn)

        x = torch.randn(1, 3, 224, 224)
        output = folded_conv(x)
        self.assertEqual(output.shape, (1, 16, 224, 224))

if __name__ == '__main__':
    unittest.main()