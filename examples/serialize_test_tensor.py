import _init_paths

import torch

from tmsg import from_torch, to_torch

# Create a torch tensor
tensor = torch.rand(1, 10, 10, 3)

# Convert the torch tensor to a flatbuffer
buf = from_torch(tensor)

# Convert the flatbuffer back to a torch tensor
tensor2, metadata = to_torch(buf)

print(tensor2.shape, tensor2.dtype)
print(metadata)
