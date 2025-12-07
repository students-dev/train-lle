import { Tensor } from "../core/tensor";
import { Layer } from "../core/engine";

export class Dense implements Layer {
  weight: Tensor;
  bias: Tensor;
  gradW: Tensor | null = null;
  gradB: Tensor | null = null;
  input: Tensor | null = null;
  output: Tensor | null = null;

  constructor(inputSize: number, outputSize: number) {
    this.weight = new Tensor(new Float32Array(inputSize * outputSize).map(() => Math.random() - 0.5), [inputSize, outputSize]);
    this.bias = new Tensor(new Float32Array(outputSize).fill(0), [outputSize]);
  }

  forward(input: Tensor): Tensor {
    this.input = input;
    // output = input @ weight + bias
    const out = input.matmul(this.weight).add(this.bias);
    this.output = out;
    return out;
  }

  backward(grad: Tensor): Tensor {
    // grad_w = input.T @ grad
    this.gradW = this.input!.transpose().matmul(grad);
    // grad_b = sum over batch
    this.gradB = new Tensor(new Float32Array(this.bias.data.length), this.bias.shape);
    const batch = grad.shape[0];
    const outSize = this.bias.data.length;
    for (let b = 0; b < batch; b++) {
      for (let o = 0; o < outSize; o++) {
        this.gradB.data[o] += grad.data[b * outSize + o];
      }
    }
    // grad_input = grad @ weight.T
    return grad.matmul(this.weight.transpose());
  }

  params(): Tensor[] {
    return [this.weight, this.bias];
  }

  grads(): Tensor[] {
    return [this.gradW!, this.gradB!];
  }
}

export class ReLU implements Layer {
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

export class Linear implements Layer {
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
  activations?: ("relu" | "sigmoid" | "linear")[];
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
      } else if (acts[i] === "sigmoid") {
        layers.push(new Sigmoid());
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