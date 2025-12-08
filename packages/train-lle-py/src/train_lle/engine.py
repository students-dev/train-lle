from .tensor import Tensor
import numpy as np

class Layer:
    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError

    def params(self):
        return []

    def grads(self):
        return []

class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input: Tensor) -> Tensor:
        out = input
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad: Tensor):
        g = grad
        for layer in reversed(self.layers):
            g = layer.backward(g)

    def params(self):
        return [p for layer in self.layers for p in layer.params()]

    def grads(self):
        return [g for layer in self.layers for g in layer.grads()]

    def predict(self, input: Tensor) -> Tensor:
        return self.forward(input)

class Trainer:
    def __init__(self, optimizer, loss, epochs):
        self.optimizer = optimizer
        self.loss_fn = loss
        self.epochs = epochs

    def fit(self, model: Model, inputs: Tensor, targets: Tensor):
        for epoch in range(self.epochs):
            pred = model.forward(inputs)
            loss = self.loss_fn.forward(pred, targets)
            grad = self.loss_fn.backward(pred, targets)
            model.backward(grad)
            self.optimizer.step(model.params(), model.grads())
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss}")

    def evaluate(self, model: Model, inputs: Tensor, targets: Tensor):
        pred = model.forward(inputs)
        return self.loss_fn.forward(pred, targets)