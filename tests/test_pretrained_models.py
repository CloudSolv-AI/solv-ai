import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from solv_ai import QuantizedLayer, PrunedLayer

def test_pretrained_model():
    # Load a pre-trained ResNet model
    model = models.resnet18(pretrained=True)
    
    # Convert model parameters to NumPy arrays for your custom layers
    def get_numpy_weights(model):
        weights = {}
        for name, param in model.named_parameters():
            weights[name] = param.data.numpy()
        return weights
    
    weights = get_numpy_weights(model)
    
    # Extract weights for a specific layer to demonstrate optimization
    # Here, we are using the first convolutional layer
    conv1_weights = weights['conv1.weight']
    
    # Initialize custom quantization and pruning layers
    quant_layer = QuantizedLayer(conv1_weights, bits=8)
    prune_layer = PrunedLayer(conv1_weights, pruning_rate=0.5)
    
    # Create dummy input data
    x = np.random.rand(1, 3, 224, 224)  # Example input shape for ResNet
    
    # Apply quantization
    quant_output = quant_layer.forward(x.reshape(1, -1))
    print("Quantized output:")
    print(quant_output)
    
    # Apply pruning
    prune_output = prune_layer.forward(x.reshape(1, -1))
    print("Pruned output:")
    print(prune_output)
    
    # Test with a specific layer's output
    # Here, the model's output before optimization is not required,
    # but you can compare results before and after applying your techniques.

if __name__ == '__main__':
    test_pretrained_model()