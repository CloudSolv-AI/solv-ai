from solv_ai import (
    QuantizedLayer, PrunedLayer, DenseLayer, MixedPrecisionLayer, EfficientNet,
    ModelParallelLayer, DataParallelLayer, PipelineParallelLayer, profile_model
)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Example usage of DenseLayer
dense = DenseLayer(10, 5)
x = np.random.rand(3, 10)
output_dense = dense.forward(x)
print("DenseLayer output:")
print(output_dense)

# Example usage of QuantizedLayer
weights = np.random.rand(5, 10)
quant_layer = QuantizedLayer(weights, bits=8)
output_quant = quant_layer.forward(x)
print("QuantizedLayer output:")
print(output_quant)

# Example usage of PrunedLayer
prune_layer = PrunedLayer(weights, pruning_rate=0.5)
output_prune = prune_layer.forward(x)
print("PrunedLayer output:")
print(output_prune)

# Example usage of MixedPrecisionLayer
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
mp_layer = MixedPrecisionLayer(model, optimizer)

x_torch = torch.randn(3, 10)
output_mp = mp_layer.forward(x_torch)
loss = output_mp.sum()
mp_layer.backward(loss)
print("MixedPrecisionLayer output:")
print(output_mp)

# Example usage of EfficientNet
model = EfficientNet()
x_torch = torch.randn(3, 3, 224, 224)
output_efficient = model(x_torch)
print("EfficientNet output:")
print(output_efficient)

# Example usage of ModelParallelLayer, DataParallelLayer, and PipelineParallelLayer
model = SimpleModel()
device_ids = [0, 1]
mp_layer = ModelParallelLayer(model, device_ids)
dp_layer = DataParallelLayer(model, device_ids)
pipeline_layer = PipelineParallelLayer([model], chunks=2)

output_mp = mp_layer(x_torch)
output_dp = dp_layer(x_torch)
output_pipeline = pipeline_layer(x_torch)
print("Model Parallel output:")
print(output_mp)
print("Data Parallel output:")
print(output_dp)
print("Pipeline Parallel output:")
print(output_pipeline)

# Example usage of profiling
profile_model(model, (3, 10))