import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from solv_ai import (
    QuantizedLayer, PrunedLayer, MixedPrecisionLayer,
    ModelParallelLayer, DataParallelLayer, CustomPipelineParallelLayer, profile_model
)

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

# Function to run the benchmark
def run_benchmark():
    # Initialize the model
    device = torch.device('cuda:0')
    # device = torch.device('cpu')
    model = SimpleModel().to(device)

    # Benchmark the original model
    print("Benchmarking original model:")
    profile_model(model, (3, 10), device)

    # Apply Quantization
    weights = np.random.rand(10, 5)  # Corrected dimensions
    quant_layer = QuantizedLayer(weights, bits=8)
    x = np.random.rand(3, 10)
    quant_output = quant_layer.forward(x)
    print("Benchmarking quantized model:")
    profile_model(model, (3, 10), device)

    # Apply Pruning
    prune_layer = PrunedLayer(weights, pruning_rate=0.5)
    prune_output = prune_layer.forward(x)
    print("Benchmarking pruned model:")
    profile_model(model, (3, 10), device)

    # Apply Mixed Precision
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mp_layer = MixedPrecisionLayer(model, optimizer)
    x_torch = torch.randn(3, 10).to(device)
    output_mp = mp_layer.forward(x_torch)
    loss = output_mp.sum()
    mp_layer.backward(loss)
    print("Benchmarking mixed precision model:")
    profile_model(model, (3, 10), device)

    # Apply EfficientNet (as an example of an efficient model)
    efficient_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).to(device)
    x_torch_efficient = torch.randn(3, 3, 224, 224).to(device)
    output_efficient = efficient_model(x_torch_efficient)
    print("Benchmarking EfficientNet model:")
    profile_model(efficient_model, (3, 3, 224, 224), device)

    # Apply Model Parallelism
    device_ids = [0]  # Use only available device IDs
    mp_layer = ModelParallelLayer(model, device_ids)
    output_mp = mp_layer(x_torch)
    print("Benchmarking model parallel model:")
    profile_model(model, (3, 10), device)

    # Apply Data Parallelism
    dp_layer = DataParallelLayer(model, device_ids)
    output_dp = dp_layer(x_torch)
    print("Benchmarking data parallel model:")
    profile_model(model, (3, 10), device)

    # Apply Custom Pipeline Parallelism
    devices = [torch.device('cuda:0')]  # Use only available device IDs
    pipeline_layer = CustomPipelineParallelLayer([model], devices)
    output_pipeline = pipeline_layer(x_torch)
    print("Benchmarking custom pipeline parallel model:")
    profile_model(pipeline_layer, (3, 10), device)

# Run the benchmark
run_benchmark()