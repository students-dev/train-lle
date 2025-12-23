import { Tensor } from "./tensor";

export class Metrics {
  static accuracy(predictions: Tensor, targets: Tensor): number {
    // predictions: [batch, classes] (logits or probs) or [batch, 1] (binary)
    // targets: [batch, classes] (one-hot) or [batch, 1] (indices or binary)
    
    const [batch, dims] = predictions.shape;
    let correct = 0;

    if (dims === 1) {
      // Binary classification
      for (let i = 0; i < batch; i++) {
        const pred = predictions.data[i] >= 0.5 ? 1 : 0;
        const target = targets.data[i];
        if (pred === target) correct++;
      }
    } else {
      // Multi-class
      // Assume targets are one-hot if dims > 1 matches preds, or indices if targets dim is 1?
      // Let's handle one-hot targets
      for (let i = 0; i < batch; i++) {
        // Find max index in pred
        let maxIdx = 0;
        let maxVal = -Infinity;
        for (let j = 0; j < dims; j++) {
          if (predictions.data[i * dims + j] > maxVal) {
            maxVal = predictions.data[i * dims + j];
            maxIdx = j;
          }
        }
        
        // Find max index in target
        let targetIdx = 0;
        let targetMax = -Infinity;
        for (let j = 0; j < dims; j++) {
            if (targets.data[i * dims + j] > targetMax) {
                targetMax = targets.data[i * dims + j];
                targetIdx = j;
            }
        }
        
        if (maxIdx === targetIdx) correct++;
      }
    }

    return correct / batch;
  }

  static mse(predictions: Tensor, targets: Tensor): number {
    let sum = 0;
    for (let i = 0; i < predictions.data.length; i++) {
      const diff = predictions.data[i] - targets.data[i];
      sum += diff * diff;
    }
    return sum / predictions.data.length;
  }
  
  static mae(predictions: Tensor, targets: Tensor): number {
    let sum = 0;
    for (let i = 0; i < predictions.data.length; i++) {
      sum += Math.abs(predictions.data[i] - targets.data[i]);
    }
    return sum / predictions.data.length;
  }
}
