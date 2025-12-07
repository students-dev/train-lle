import { Tensor } from "./tensor";

export interface Loss {
  forward(pred: Tensor, target: Tensor): number;
  backward(pred: Tensor, target: Tensor): Tensor;
}

/**
 * Mean Squared Error loss.
 */
export class MSE implements Loss {
  forward(pred: Tensor, target: Tensor): number {
    let sum = 0;
    for (let i = 0; i < pred.data.length; i++) {
      const diff = pred.data[i] - target.data[i];
      sum += diff * diff;
    }
    return sum / pred.data.length;
  }

  backward(pred: Tensor, target: Tensor): Tensor {
    const grad = new Float32Array(pred.data.length);
    for (let i = 0; i < pred.data.length; i++) {
      grad[i] = 2 * (pred.data[i] - target.data[i]) / pred.data.length;
    }
    return new Tensor(grad, pred.shape);
  }
}

/**
 * Cross-Entropy loss with softmax.
 */
export class CrossEntropy implements Loss {
  forward(pred: Tensor, target: Tensor): number {
    // Assume pred is logits, target is one-hot or class indices
    // For simplicity, assume target is class indices, pred is [batch, classes]
    const batch = pred.shape[0];
    const classes = pred.shape[1];
    let loss = 0;
    for (let b = 0; b < batch; b++) {
      // Softmax
      let maxLogit = -Infinity;
      for (let c = 0; c < classes; c++) {
        maxLogit = Math.max(maxLogit, pred.data[b * classes + c]);
      }
      let sumExp = 0;
      for (let c = 0; c < classes; c++) {
        sumExp += Math.exp(pred.data[b * classes + c] - maxLogit);
      }
      const logSumExp = maxLogit + Math.log(sumExp);
      const targetClass = target.data[b];
      loss += logSumExp - pred.data[b * classes + targetClass];
    }
    return loss / batch;
  }

  backward(pred: Tensor, target: Tensor): Tensor {
    // Grad = softmax(pred) - target (one-hot)
    const batch = pred.shape[0];
    const classes = pred.shape[1];
    const grad = new Float32Array(pred.data.length);
    for (let b = 0; b < batch; b++) {
      // Softmax
      let maxLogit = -Infinity;
      for (let c = 0; c < classes; c++) {
        maxLogit = Math.max(maxLogit, pred.data[b * classes + c]);
      }
      let sumExp = 0;
      const exps = new Float32Array(classes);
      for (let c = 0; c < classes; c++) {
        exps[c] = Math.exp(pred.data[b * classes + c] - maxLogit);
        sumExp += exps[c];
      }
      for (let c = 0; c < classes; c++) {
        const softmax = exps[c] / sumExp;
        const targetVal = c === target.data[b] ? 1 : 0;
        grad[b * classes + c] = softmax - targetVal;
      }
    }
    return new Tensor(grad, pred.shape).mul(1 / batch);
  }
}