import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from solv_ai import (
    QuantizedLayer, PrunedLayer, MixedPrecisionLayer,
    ModelParallelLayer, DataParallelLayer, CustomPipelineParallelLayer, profile_model
)
import time

# Enable CUDA launch blocking for better debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Enable device-side assertions
os.environ['TORCH_USE_CUDA_DSA'] = '1'
# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Load the local model
class LocalModel(nn.Module):
    def __init__(self):
        super(LocalModel, self).__init__()
        # Load the model from the .pth file
        self.model = LlavaForConditionalGeneration.from_pretrained(r'C:\Users\Eric\Desktop\llava')
        self.model.load_state_dict(torch.load(r'C:\Users\Eric\Desktop\llava\model.pth'))
        self.model.to(torch.device('cuda:0'))
        self.tokenizer = LlavaProcessor.from_pretrained(r'C:\Users\Eric\Desktop\llava')  # Load the tokenizer

        # Manually add special tokens to the tokenizer
        special_tokens_dict = {'additional_special_tokens': ['<special1>', '<special2>']}
        num_added_toks = self.tokenizer.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer.tokenizer))

    def forward(self, x):
        x = x.long()  # Ensure input is of type LongTensor
        print(f"Input to model: {x}")
        return self.model(x)

def run_benchmark():
    device = torch.device('cuda:0')
    model = LocalModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Generate some random input data using the tokenizer
    sentences = ["Hello, how are you?", "What is your name?", "Tell me a joke."]
    tokenizer = model.tokenizer
    input_data = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=128).input_ids.to(device)
    print(f"Input data: {input_data}")
    print(f"Input data min: {input_data.min().item()}, max: {input_data.max().item()}")

    # Ensure input data is within the valid range of the model's vocabulary
    vocab_size = len(tokenizer.tokenizer)  # Adjusted vocabulary size
    if input_data.max().item() >= vocab_size:
        raise ValueError(f"Token ID {input_data.max().item()} is out of the valid range (0, {vocab_size-1})")

    # Benchmark the original model
    print("Benchmarking original model:")
    start_time = time.perf_counter()
    profile_model(model, input_data.size(), device)
    end_time = time.perf_counter()
    print(f"Original model inference time: {end_time - start_time:.10f} seconds")

    # Apply Mixed Precision
    mp_layer = MixedPrecisionLayer(model, optimizer)
    x_torch = torch.randn(3, 10).to(device)
    output_mp = mp_layer.forward(x_torch)
    loss = output_mp.sum()
    mp_layer.backward(loss)

    # Apply Model Parallelism
    device_ids = [0]  # Use only available device IDs
    mp_layer = ModelParallelLayer(model, device_ids)
    output_mp = mp_layer(x_torch)

    # Apply Data Parallelism
    dp_layer = DataParallelLayer(model, device_ids)
    output_dp = dp_layer(x_torch)

    # Apply Custom Pipeline Parallelism
    devices = [torch.device('cuda:0')]  # Use only available device IDs
    pipeline_layer = CustomPipelineParallelLayer([model], devices)
    output_pipeline = pipeline_layer(x_torch)

    # Benchmark the optimized model
    print("Benchmarking optimized model:")
    start_time = time.perf_counter()
    profile_model(pipeline_layer, input_data.size(), device)
    end_time = time.perf_counter()
    print(f"Optimized model inference time: {end_time - start_time:.10f} seconds")

# Run the benchmark
run_benchmark()