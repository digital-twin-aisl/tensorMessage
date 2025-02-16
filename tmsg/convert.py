# built-in dependencies
from typing import Tuple

# 3rd party dependencies
import flatbuffers
import numpy as np
import torch

# Local dependencies
from tmsg.generated.HeatmapSchema import Heatmap

torch_type_map = {
    'torch.float16': torch.float16,
    'torch.float32': torch.float32,
    'torch.float64': torch.float64,
    'torch.int8': torch.int8,
    'torch.int16': torch.int16,
    'torch.int32': torch.int32,
    'torch.int64': torch.int64,
    'torch.uint8': torch.uint8,
    'torch.uint16': torch.uint16,
    'torch.uint32': torch.uint32,
    'torch.uint64': torch.uint64,
    'torch.bool': torch.bool,
}


def from_numpy(array: np.ndarray) -> bytes:
    """
    Convert a numpy array to a flatbuffer Heatmap object.

    Parameters
    ----------
    array : np.ndarray
        The numpy array to convert.

    Returns
    -------
    bytes
        The byte representation of the flatbuffer Heatmap object.
    """
    # Create a new flatbuffer builder
    builder = flatbuffers.Builder(1024)

    # NumPy to Bytes
    data_bytes = array.tobytes()
    data_vector = builder.CreateByteVector(data_bytes)

    # String to Bytes
    type_str = builder.CreateString(str(array.dtype))

    # Create the flatbuffer Heatmap object
    Heatmap.Start(builder)
    Heatmap.AddBatch(builder, 0)
    Heatmap.AddWidth(builder, array.shape[0])
    Heatmap.AddHeight(builder, array.shape[1])
    Heatmap.AddChannels(builder, array.shape[2])
    Heatmap.AddType(builder, type_str)
    Heatmap.AddData(builder, data_vector)
    heatmap = Heatmap.End(builder)

    builder.Finish(heatmap)
    
    return builder.Output()

def to_numpy(buf: bytes) -> Tuple[np.ndarray, dict]:
    """
    Convert a flatbuffer Heatmap object to a numpy array.

    Parameters
    ----------
    buf : bytes
        The byte representation of the flatbuffer Heatmap object.

    Returns
    -------
    np.ndarray
        The numpy array representation of the flatbuffer Heatmap object.
    dict
        Metadata containing the shape and dtype of the numpy array.
    """
    # Deserialize the flatbuffer
    heatmap = Heatmap.Heatmap.GetRootAsHeatmap(buf, 0)
    
    # Read Metadata
    width = heatmap.Width()
    height = heatmap.Height()
    channels = heatmap.Channels()
    dtype = heatmap.Type().decode('utf-8')
    
    # Create Metadata
    shape = (width, height, channels)
    metadata = {'shape': shape, 'dtype': dtype}
    
    # Read Data
    data_bytes = heatmap.DataAsNumpy()
    
    # Convert Data to Numpy Array
    array = np.frombuffer(data_bytes, dtype=dtype)
    array = array.reshape((width, height, channels))
    
    return array, metadata


def from_torch(tensor: torch.Tensor) -> bytes:
    """
    Convert a torch tensor to a flatbuffer Heatmap object.

    Parameters
    ----------
    tensor : torch.Tensor
        The torch tensor to convert. The tensor should have a shape of (batch, channels, height, width).

    Returns
    -------
    bytes
        The byte representation of the flatbuffer Heatmap object.
    """
    # Create a new flatbuffer builder
    builder = flatbuffers.Builder(1024)
    
    # Torch to bytes
    batch, channels, height, width = tensor.shape
    array = tensor.cpu().numpy()
    dtype = array.dtype
    data_bytes = array.astype(dtype).tobytes()
    
    # String to Bytes
    type_str = builder.CreateString(str(dtype))
    
    # Create the flatbuffer Heatmap object
    Heatmap.Start(builder)
    Heatmap.AddBatch(builder, batch)
    Heatmap.AddWidth(builder, width)
    Heatmap.AddHeight(builder, height)
    Heatmap.AddChannels(builder, channels)
    Heatmap.AddType(builder, type_str)
    Heatmap.AddData(builder, data_bytes)
    heatmap = Heatmap.End(builder)
    
    builder.Finish(heatmap)
    
    return builder.Output()

def to_torch(buf: bytes) -> Tuple[torch.Tensor, dict]:
    """
    Convert a flatbuffer Heatmap object to a torch tensor.

    Parameters
    ----------
    buf : bytes
        The byte representation of the flatbuffer Heatmap object.

    Returns
    -------
    torch.Tensor
        The torch tensor representation of the flatbuffer Heatmap object.
    dict
        Metadata containing the shape and dtype of the torch tensor.
    """
    # Deserialize the flatbuffer
    heatmap = Heatmap.Heatmap.GetRootAsHeatmap(buf, 0)
    
    # Read Metadata
    batch = heatmap.Batch()
    width = heatmap.Width()
    height = heatmap.Height()
    channels = heatmap.Channels()
    dtype = heatmap.Type().decode('utf-8')
    
    # Create Metadata
    shape = (batch, width, height, channels)
    metadata = {'shape': shape, 'dtype': dtype}
    
    # Read Data
    data_bytes = heatmap.DataAsNumpy()
    
    # Convert Data to Torch Tensor
    tensor = torch.from_numpy(np.frombuffer(data_bytes, dtype=dtype))
    tensor = tensor.view(shape)
    
    return tensor, metadata
