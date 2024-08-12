# Install PyTorch with CUDA support
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import torch

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Print the name of the GPU
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))