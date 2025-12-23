import numpy as np

class Tensor:
    def __init__(self, data, shape=None):
        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        elif not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        if shape is not None:
            data = data.reshape(shape)
        self.data = data.astype(np.float32)
        self.shape = tuple(self.data.shape)
        self.grad = None

    def __add__(self, other):
        return Tensor(self.data + other.data)

    def __sub__(self, other):
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self.data * other)
        return Tensor(self.data * other.data)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self.data / other)
        return Tensor(self.data / other.data)

    def __matmul__(self, other):
        return Tensor(self.data @ other.data)

    def sum(self, axis=None):
        return Tensor(self.data.sum(axis=axis, keepdims=True))

    def mean(self, axis=None):
        return Tensor(self.data.mean(axis=axis, keepdims=True))

    def exp(self):
        return Tensor(np.exp(self.data))

    def log(self):
        return Tensor(np.log(self.data))

    def sqrt(self):
        return Tensor(np.sqrt(self.data))

    def tanh(self):
        return Tensor(np.tanh(self.data))

    def relu(self):
        return Tensor(np.maximum(0, self.data))

    def sigmoid(self):
        return Tensor(1 / (1 + np.exp(-self.data)))

    def softmax(self):
        exp = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        return Tensor(exp / np.sum(exp, axis=-1, keepdims=True))

    def transpose(self):
        return Tensor(self.data.T)

    def reshape(self, shape):
        return Tensor(self.data.reshape(shape))

    def __repr__(self):
        return f"Tensor({self.data}, shape={self.shape})"