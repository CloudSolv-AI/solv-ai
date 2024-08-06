import torch
import torch.nn as nn
import unittest
from solv_ai import ModelParallelLayer, DataParallelLayer, PipelineParallelLayer

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

class TestDistributedComputing(unittest.TestCase):
    def test_model_parallel_layer(self):
        model = SimpleModel()
        device_ids = [0, 1]
        mp_layer = ModelParallelLayer(model, device_ids)
        x = torch.randn(3, 10)
        output = mp_layer(x)
        self.assertEqual(output.shape, (3, 5))

    def test_data_parallel_layer(self):
        model = SimpleModel()
        device_ids = [0, 1]
        dp_layer = DataParallelLayer(model, device_ids)
        x = torch.randn(3, 10)
        output = dp_layer(x)
        self.assertEqual(output.shape, (3, 5))

    def test_pipeline_parallel_layer(self):
        model = SimpleModel()
        pipeline_layer = PipelineParallelLayer([model], chunks=2)
        x = torch.randn(3, 10)
        output = pipeline_layer(x)
        self.assertEqual(output.shape, (3, 5))

if __name__ == '__main__':
    unittest.main()