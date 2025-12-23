from .tensor import Tensor
from .engine import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weight = Tensor(np.random.randn(input_size, output_size) * 0.1)
        self.bias = Tensor(np.zeros(output_size))
        self.grad_w = None
        self.grad_b = None
        self.input = None

    def forward(self, input):
        self.input = input
        return input @ self.weight + self.bias

    def backward(self, grad):
        self.grad_w = self.input.transpose() @ grad
        self.grad_b = grad.sum(axis=0)
        return grad @ self.weight.transpose()

    def params(self):
        return [self.weight, self.bias]

    def grads(self):
        return [self.grad_w, self.grad_b]

class ReLU(Layer):
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        return input.relu()

    def backward(self, grad):
        mask = (self.input.data > 0).astype(np.float32)
        return Tensor(grad.data * mask, grad.shape)

    def params(self):
        return []

    def grads(self):
        return []

class MLP:
    @staticmethod
    def build(config):
        layers = []
        prev = config['input']
        activations = config.get('activations', ['relu'] * len(config['layers']))
        for i, hidden in enumerate(config['layers']):
            layers.append(Dense(prev, hidden))
            if activations[i] == 'relu':
                layers.append(ReLU())
            prev = hidden
        layers.append(Dense(prev, config['output']))
        return layers