import torch
from transformers import LlavaForConditionalGeneration

# Load the model from the safetensors file
model = LlavaForConditionalGeneration.from_pretrained(r'C:\Users\Eric\Desktop\llava', use_safetensors=True)

# Save the model's state dictionary to a .pth file
torch.save(model.state_dict(), r'C:\Users\Eric\Desktop\llava\model.pth')

print("Model has been successfully converted to model.pth")