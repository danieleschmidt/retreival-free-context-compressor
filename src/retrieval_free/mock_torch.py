"""
Mock PyTorch module for demonstration purposes.
This allows the system to run without requiring the full PyTorch installation.
"""

import numpy as np
from typing import Any, List, Dict, Optional, Union


class MockTensor:
    """Mock tensor class that mimics basic PyTorch tensor functionality."""
    
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)
    
    def numpy(self):
        return self.data
    
    def to(self, device):
        return self
    
    def cuda(self):
        return self
    
    def cpu(self):
        return self
    
    @property
    def shape(self):
        return self.data.shape
    
    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]


def tensor(data):
    """Create a mock tensor."""
    return MockTensor(data)


class MockCuda:
    """Mock CUDA module."""
    
    @staticmethod
    def is_available():
        return False
    
    @staticmethod
    def device_count():
        return 0


cuda = MockCuda()


class MockDevice:
    """Mock device class."""
    
    def __init__(self, device_string="cpu"):
        self.device_string = device_string
    
    def __str__(self):
        return self.device_string


def device(device_string):
    """Create a mock device."""
    return MockDevice(device_string)


class MockNN:
    """Mock neural network module."""
    
    class Module:
        """Mock module base class."""
        
        def __init__(self):
            pass
        
        def to(self, device):
            return self
        
        def eval(self):
            return self
        
        def train(self):
            return self


nn = MockNN()


# Mock functional operations
class MockF:
    """Mock functional operations."""
    
    @staticmethod
    def cosine_similarity(x1, x2, dim=1):
        """Mock cosine similarity."""
        # Convert to numpy if needed
        if hasattr(x1, 'numpy'):
            x1 = x1.numpy()
        if hasattr(x2, 'numpy'):
            x2 = x2.numpy()
        
        # Compute cosine similarity
        dot_product = np.sum(x1 * x2, axis=dim, keepdims=True)
        norm_x1 = np.linalg.norm(x1, axis=dim, keepdims=True)
        norm_x2 = np.linalg.norm(x2, axis=dim, keepdims=True)
        
        similarity = dot_product / (norm_x1 * norm_x2 + 1e-8)
        return MockTensor(similarity)


F = MockF()


# Mock no_grad context manager
class NoGrad:
    """Mock no_grad context manager."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def no_grad():
    """Mock no_grad context manager."""
    return NoGrad()


# Mock functions that are commonly used
def stack(tensors, dim=0):
    """Mock tensor stacking."""
    arrays = [t.data if hasattr(t, 'data') else t for t in tensors]
    stacked = np.stack(arrays, axis=dim)
    return MockTensor(stacked)


def cat(tensors, dim=0):
    """Mock tensor concatenation."""
    arrays = [t.data if hasattr(t, 'data') else t for t in tensors]
    concatenated = np.concatenate(arrays, axis=dim)
    return MockTensor(concatenated)


def mean(tensor, dim=None, keepdim=False):
    """Mock tensor mean."""
    data = tensor.data if hasattr(tensor, 'data') else tensor
    result = np.mean(data, axis=dim, keepdims=keepdim)
    return MockTensor(result)


def sum(tensor, dim=None, keepdim=False):
    """Mock tensor sum."""
    data = tensor.data if hasattr(tensor, 'data') else tensor
    result = np.sum(data, axis=dim, keepdims=keepdim)
    return MockTensor(result)


def zeros(size, dtype=None, device=None):
    """Mock zeros tensor creation."""
    return MockTensor(np.zeros(size))


def ones(size, dtype=None, device=None):
    """Mock ones tensor creation."""
    return MockTensor(np.ones(size))


def randn(*size, dtype=None, device=None):
    """Mock random normal tensor creation."""
    return MockTensor(np.random.randn(*size))


# Mock dtype
float32 = np.float32
float64 = np.float64