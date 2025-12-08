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
    } else {
      const out = input.matmul(this.weight).add(this.bias);
      this.output = out;
      return out;
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