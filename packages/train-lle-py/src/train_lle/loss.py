from .tensor import Tensor
import numpy as np

class Loss:
    def forward(self, pred, target):
        raise NotImplementedError

    def backward(self, pred, target):
        raise NotImplementedError

class MSE(Loss):
    def forward(self, pred, target):
        diff = pred.data - target.data
        return np.mean(diff ** 2)

    def backward(self, pred, target):
        diff = pred.data - target.data
        grad = 2 * diff / pred.data.size
        return Tensor(grad, pred.shape)

class CrossEntropy(Loss):
    def forward(self, pred, target):
        # Assume pred is logits [batch, classes], target is class indices [batch]
        batch, classes = pred.shape
        loss = 0
        for b in range(batch):
            logits = pred.data[b]
            max_logit = np.max(logits)
            log_sum_exp = max_logit + np.log(np.sum(np.exp(logits - max_logit)))
            target_class = int(target.data[b])
            loss += log_sum_exp - logits[target_class]
        return loss / batch

    def backward(self, pred, target):
        # Grad = softmax(pred) - one_hot(target)
        batch, classes = pred.shape
        grad = np.zeros_like(pred.data)
        for b in range(batch):
            logits = pred.data[b]
            max_logit = np.max(logits)
            exps = np.exp(logits - max_logit)
            softmax = exps / np.sum(exps)
            target_class = int(target.data[b])
            one_hot = np.zeros(classes)
            one_hot[target_class] = 1
            grad[b] = softmax - one_hot
        return Tensor(grad / batch, pred.shape)