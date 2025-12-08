from .tensor import Tensor
import numpy as np

class Optimizer:
    def step(self, params, grads):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        for i in range(len(params)):
            params[i] = params[i] + grads[i] * (-self.lr)

class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = []
        self.v = []

    def step(self, params, grads):
        self.t += 1
        if not self.m:
            self.m = [Tensor(np.zeros_like(g.data), g.shape) for g in grads]
            self.v = [Tensor(np.zeros_like(g.data), g.shape) for g in grads]
        for i in range(len(params)):
            # m = beta1 * m + (1 - beta1) * grad
            self.m[i] = self.m[i] * self.beta1 + grads[i] * (1 - self.beta1)
            # v = beta2 * v + (1 - beta2) * grad^2
            self.v[i] = self.v[i] * self.beta2 + (grads[i] * grads[i]) * (1 - self.beta2)
            # m_hat = m / (1 - beta1^t)
            m_hat = self.m[i] * (1 / (1 - self.beta1 ** self.t))
            # v_hat = v / (1 - beta2^t)
            v_hat = self.v[i] * (1 / (1 - self.beta2 ** self.t))
            # params -= lr * m_hat / (sqrt(v_hat) + epsilon)
            denom = v_hat.sqrt() + self.epsilon
            update = m_hat * self.lr / denom
            params[i] = params[i] - update