import { Tensor } from "../core/tensor";
import { Layer } from "../core/engine";

export class Dense implements Layer {
  type = "dense";
  config: { inputSize: number; outputSize: number };
  weight: Tensor;
  bias: Tensor;
  gradW: Tensor | null = null;
  gradB: Tensor | null = null;
  input: Tensor | null = null;
  output: Tensor | null = null;

  constructor(inputSize: number, outputSize: number) {
    this.config = { inputSize, outputSize };
    this.weight = new Tensor(new Float32Array(inputSize * outputSize).map(() => Math.random() - 0.5), [inputSize, outputSize]);
    this.bias = new Tensor(new Float32Array(outputSize).fill(0), [outputSize]);
  }

  forward(input: Tensor): Tensor {
    this.input = input;
    if (input.shape.length === 1) {
      // Treat as batch 1
      const batchedInput = new Tensor(input.data, [1, input.shape[0]]);
      const out = batchedInput.matmul(this.weight);
      const biasBatched = new Tensor(this.bias.data, [1, this.bias.shape[0]]);
      const outWithBias = out.add(biasBatched);
      this.output = new Tensor(outWithBias.data, [outWithBias.shape[1]]);
      return this.output;
    } else if (input.shape.length === 2) {
      const out = input.matmul(this.weight).add(this.bias);
      this.output = out;
      return out;
    } else {
      // Handle > 2 dimensions (e.g. [batch, seq, features])
      // Collapse all leading dims
      const lastDim = input.shape[input.shape.length - 1];
      const leadingDims = input.shape.slice(0, -1);
      const totalBatch = leadingDims.reduce((a, b) => a * b, 1);
      
      const flattenedInput = new Tensor(input.data, [totalBatch, lastDim]);
      const outFlat = flattenedInput.matmul(this.weight).add(this.bias);
      
      // Reshape back
      const newShape = [...leadingDims, this.config.outputSize];
      this.output = new Tensor(outFlat.data, newShape);
      return this.output;
    }
  }

  backward(grad: Tensor): Tensor {
    if (this.input!.shape.length === 1) {
      // 1D
      const batchedInput = new Tensor(this.input!.data, [1, this.input!.shape[0]]);
      const batchedGrad = new Tensor(grad.data, [1, grad.shape[0]]);
      this.gradW = batchedInput.transpose().matmul(batchedGrad);
      this.gradB = batchedGrad.sum(0);
      const dInputBatched = batchedGrad.matmul(this.weight.transpose());
      return new Tensor(dInputBatched.data, [dInputBatched.shape[1]]);
    } else {
      // batched
      this.gradW = this.input!.transpose().matmul(grad);
      this.gradB = grad.sum(0);
      return grad.matmul(this.weight.transpose());
    }
  }

  params(): Tensor[] {
    return [this.weight, this.bias];
  }

  grads(): Tensor[] {
    return [this.gradW!, this.gradB!];
  }
}

export class ReLU implements Layer {
  type = "relu";
  config = {};
  input: Tensor | null = null;

  forward(input: Tensor): Tensor {
    this.input = input;
    const out = new Float32Array(input.data.length);
    for (let i = 0; i < input.data.length; i++) {
      out[i] = Math.max(0, input.data[i]);
    }
    return new Tensor(out, input.shape);
  }

  backward(grad: Tensor): Tensor {
    const out = new Float32Array(grad.data.length);
    for (let i = 0; i < grad.data.length; i++) {
      out[i] = this.input!.data[i] > 0 ? grad.data[i] : 0;
    }
    return new Tensor(out, grad.shape);
  }

  params(): Tensor[] {
    return [];
  }

  grads(): Tensor[] {
    return [];
  }
}

export class Sigmoid implements Layer {
  type = "sigmoid";
  config = {};
  input: Tensor | null = null;

  forward(input: Tensor): Tensor {
    this.input = input;
    const out = new Float32Array(input.data.length);
    for (let i = 0; i < input.data.length; i++) {
      out[i] = 1 / (1 + Math.exp(-input.data[i]));
    }
    return new Tensor(out, input.shape);
  }

  backward(grad: Tensor): Tensor {
    const out = new Float32Array(grad.data.length);
    for (let i = 0; i < grad.data.length; i++) {
      const sig = 1 / (1 + Math.exp(-this.input!.data[i]));
      out[i] = grad.data[i] * sig * (1 - sig);
    }
    return new Tensor(out, grad.shape);
  }

  params(): Tensor[] {
    return [];
  }

  grads(): Tensor[] {
    return [];
  }
}

export class Tanh implements Layer {
  type = "tanh";
  config = {};
  input: Tensor | null = null;

  forward(input: Tensor): Tensor {
    this.input = input;
    const out = new Float32Array(input.data.length);
    for (let i = 0; i < input.data.length; i++) {
      out[i] = Math.tanh(input.data[i]);
    }
    return new Tensor(out, input.shape);
  }

  backward(grad: Tensor): Tensor {
    const out = new Float32Array(grad.data.length);
    for (let i = 0; i < grad.data.length; i++) {
      const t = Math.tanh(this.input!.data[i]);
      out[i] = grad.data[i] * (1 - t * t);
    }
    return new Tensor(out, grad.shape);
  }

  params(): Tensor[] {
    return [];
  }

  grads(): Tensor[] {
    return [];
  }
}

export class Softmax implements Layer {
  type = "softmax";
  config = {};
  input: Tensor | null = null;
  output: Tensor | null = null;

  forward(input: Tensor): Tensor {
    this.input = input;
    // Assume last dim is classes
    const [batch, ...dims] = input.shape;
    const numClasses = dims[dims.length - 1];
    const out = new Float32Array(input.data.length);
    for (let b = 0; b < batch; b++) {
      const start = b * numClasses;
      let max = -Infinity;
      for (let c = 0; c < numClasses; c++) {
        max = Math.max(max, input.data[start + c]);
      }
      let sum = 0;
      for (let c = 0; c < numClasses; c++) {
        out[start + c] = Math.exp(input.data[start + c] - max);
        sum += out[start + c];
      }
      for (let c = 0; c < numClasses; c++) {
        out[start + c] /= sum;
      }
    }
    this.output = new Tensor(out, input.shape);
    return this.output;
  }

  backward(grad: Tensor): Tensor {
    // Simplified, assume cross entropy loss
    return grad; // Placeholder
  }

  params(): Tensor[] {
    return [];
  }

  grads(): Tensor[] {
    return [];
  }
}

export class Linear implements Layer {
  type = "linear";
  config = {};

  forward(input: Tensor): Tensor {
    return input;
  }

  backward(grad: Tensor): Tensor {
    return grad;
  }

  params(): Tensor[] {
    return [];
  }

  grads(): Tensor[] {
    return [];
  }
}

export class Dropout implements Layer {
  type = "dropout";
  config: { rate: number };
  training = true;
  mask: Tensor | null = null;

  constructor(rate: number) {
    this.config = { rate };
  }

  forward(input: Tensor): Tensor {
    if (!this.training) {
      return input;
    }
    const maskData = new Float32Array(input.data.length);
    const scale = 1 / (1 - this.config.rate);
    for (let i = 0; i < input.data.length; i++) {
      maskData[i] = Math.random() > this.config.rate ? scale : 0;
    }
    this.mask = new Tensor(maskData, input.shape);
    const outData = new Float32Array(input.data.length);
    for (let i = 0; i < input.data.length; i++) {
      outData[i] = input.data[i] * maskData[i];
    }
    return new Tensor(outData, input.shape);
  }

  backward(grad: Tensor): Tensor {
    if (!this.training || !this.mask) {
      return grad;
    }
    const outData = new Float32Array(grad.data.length);
    for (let i = 0; i < grad.data.length; i++) {
      outData[i] = grad.data[i] * this.mask.data[i];
    }
    return new Tensor(outData, grad.shape);
  }

  params(): Tensor[] {
    return [];
  }

  grads(): Tensor[] {
    return [];
  }
}

export class BatchNormalization implements Layer {
  type = "batchnorm";
  config: { epsilon: number; momentum: number };
  training = true;
  gamma: Tensor;
  beta: Tensor;
  gradGamma: Tensor | null = null;
  gradBeta: Tensor | null = null;

  runningMean: Tensor;
  runningVar: Tensor;

  private xHat: Tensor | null = null;
  private stdInv: Tensor | null = null;
  private inputCentered: Tensor | null = null;

  constructor(size: number, epsilon = 1e-5, momentum = 0.9) {
    this.config = { epsilon, momentum };
    this.gamma = new Tensor(new Float32Array(size).fill(1), [size]);
    this.beta = new Tensor(new Float32Array(size).fill(0), [size]);
    this.runningMean = new Tensor(new Float32Array(size).fill(0), [size]);
    this.runningVar = new Tensor(new Float32Array(size).fill(1), [size]);
  }

  forward(input: Tensor): Tensor {
    const [batchSize, features] = input.shape.length === 1 ? [1, input.shape[0]] : [input.shape[0], input.shape[1]];
    const outData = new Float32Array(input.data.length);

    if (this.training) {
      // Compute mean and variance for current batch
      const mean = new Float32Array(features);
      const variance = new Float32Array(features);

      for (let f = 0; f < features; f++) {
        let sum = 0;
        for (let b = 0; b < batchSize; b++) {
          sum += input.data[b * features + f];
        }
        mean[f] = sum / batchSize;

        let sumSq = 0;
        for (let b = 0; b < batchSize; b++) {
          const diff = input.data[b * features + f] - mean[f];
          sumSq += diff * diff;
        }
        variance[f] = sumSq / batchSize;

        // Update running stats
        this.runningMean.data[f] = this.config.momentum * this.runningMean.data[f] + (1 - this.config.momentum) * mean[f];
        this.runningVar.data[f] = this.config.momentum * this.runningVar.data[f] + (1 - this.config.momentum) * variance[f];
      }

      this.stdInv = new Tensor(variance.map(v => 1 / Math.sqrt(v + this.config.epsilon)), [features]);
      this.inputCentered = new Tensor(new Float32Array(input.data.length), input.shape);
      this.xHat = new Tensor(new Float32Array(input.data.length), input.shape);

      for (let b = 0; b < batchSize; b++) {
        for (let f = 0; f < features; f++) {
          const idx = b * features + f;
          this.inputCentered.data[idx] = input.data[idx] - mean[f];
          this.xHat.data[idx] = this.inputCentered.data[idx] * this.stdInv.data[f];
          outData[idx] = this.xHat.data[idx] * this.gamma.data[f] + this.beta.data[f];
        }
      }
    } else {
      for (let b = 0; b < batchSize; b++) {
        for (let f = 0; f < features; f++) {
          const idx = b * features + f;
          const xHat = (input.data[idx] - this.runningMean.data[f]) / Math.sqrt(this.runningVar.data[f] + this.config.epsilon);
          outData[idx] = xHat * this.gamma.data[f] + this.beta.data[f];
        }
      }
    }

    return new Tensor(outData, input.shape);
  }

  backward(grad: Tensor): Tensor {
    const [batchSize, features] = grad.shape.length === 1 ? [1, grad.shape[0]] : [grad.shape[0], grad.shape[1]];
    const dInput = new Float32Array(grad.data.length);
    this.gradGamma = new Tensor(new Float32Array(features), [features]);
    this.gradBeta = new Tensor(new Float32Array(features), [features]);

    for (let f = 0; f < features; f++) {
      let dGammaSum = 0;
      let dBetaSum = 0;
      for (let b = 0; b < batchSize; b++) {
        const idx = b * features + f;
        dGammaSum += grad.data[idx] * this.xHat!.data[idx];
        dBetaSum += grad.data[idx];
      }
      this.gradGamma.data[f] = dGammaSum;
      this.gradBeta.data[f] = dBetaSum;

      for (let b = 0; b < batchSize; b++) {
        const idx = b * features + f;
        // Backprop through batch norm
        const dxHat = grad.data[idx] * this.gamma.data[f];
        dInput[idx] = (1 / batchSize) * this.stdInv!.data[f] * (
          batchSize * dxHat - this.gradBeta.data[f] - this.xHat!.data[idx] * this.gradGamma.data[f]
        );
      }
    }

    return new Tensor(dInput, grad.shape);
  }

  params(): Tensor[] {
    return [this.gamma, this.beta];
  }

  grads(): Tensor[] {
    return [this.gradGamma!, this.gradBeta!];
  }
}

export interface MLPConfig {
  input: number;
  layers: number[];
  output: number;
  activations?: ("relu" | "linear")[];
}

export class MLP {
  static build(config: MLPConfig): Layer[] {
    const layers: Layer[] = [];
    let prev = config.input;
    const acts = config.activations || config.layers.map(() => "relu");
    for (let i = 0; i < config.layers.length; i++) {
      layers.push(new Dense(prev, config.layers[i]));
      if (acts[i] === "relu") {
        layers.push(new ReLU());
      } else {
        layers.push(new Linear());
      }
      prev = config.layers[i];
    }
    layers.push(new Dense(prev, config.output));
    layers.push(new Linear()); // output activation
    return layers;
  }
}