import _init_paths

import numpy as np

from tmsg import from_numpy, to_numpy

# Create a numpy array
array = np.random.rand(10, 10, 3)

# Convert the numpy array to a flatbuffer
buf = from_numpy(array)

# Convert the flatbuffer back to a numpy array
array, metadata = to_numpy(buf)

print(array.shape, array.dtype)
print(metadata)
