import { Tensor } from "../core/tensor";
import { Layer } from "../core/engine";
import { Dense } from "./mlp";

export interface RNNConfig {
  inputSize: number;
  hiddenSize: number;
  outputSize: number;
}

export class RNN implements Layer {
  type = "rnn";
  config: { inputSize: number; hiddenSize: number };
  Wxh: Tensor; // [inputSize, hiddenSize]
  Whh: Tensor; // [hiddenSize, hiddenSize]
  bh: Tensor; // [hiddenSize]
  gradWxh: Tensor | null = null;
  gradWhh: Tensor | null = null;
  gradBh: Tensor | null = null;
  inputs: Tensor[] = [];
  hs: Tensor[] = [];
  training: boolean = true;

  constructor(inputSize: number, hiddenSize: number) {
    this.config = { inputSize, hiddenSize };
    this.Wxh = new Tensor(new Float32Array(inputSize * hiddenSize).map(() => Math.random() - 0.5), [inputSize, hiddenSize]);
    this.Whh = new Tensor(new Float32Array(hiddenSize * hiddenSize).map(() => Math.random() - 0.5), [hiddenSize, hiddenSize]);
    this.bh = new Tensor(new Float32Array(hiddenSize).fill(0), [hiddenSize]);
  }

  forward(input: Tensor): Tensor {
    // Assume input [batch, seqLen, inputSize]
    const [batch, seqLen, inputSize] = input.shape;
    const hiddenSize = this.Wxh.shape[1];
    const outputs: Tensor[] = [];
    let h = new Tensor(new Float32Array(batch * hiddenSize).fill(0), [batch, hiddenSize]); // initial h
    this.inputs = [];
    this.hs = [h];
    for (let t = 0; t < seqLen; t++) {
      const x = input.slice(batch * t * inputSize, batch * (t + 1) * inputSize); // [batch, inputSize]
      // h = tanh(x @ Wxh + h @ Whh + bh)
      const xWxh = x.matmul(this.Wxh);
      const hWhh = h.matmul(this.Whh);
      h = xWxh.add(hWhh).add(this.bh).tanh();
      outputs.push(h);
      this.inputs.push(x);
      this.hs.push(h);
    }
    // Stack outputs [batch, seqLen, hiddenSize]
    const outData = new Float32Array(batch * seqLen * hiddenSize);
    for (let t = 0; t < seqLen; t++) {
      for (let i = 0; i < outputs[t].data.length; i++) {
        outData[t * batch * hiddenSize + i] = outputs[t].data[i];
      }
    }
    return new Tensor(outData, [batch, seqLen, hiddenSize]);
  }

  backward(grad: Tensor): Tensor {
    // Simplified backprop for RNN
    const [batch, seqLen, hiddenSize] = grad.shape;
    const inputSize = this.Wxh.shape[0];
    const gradIn = new Float32Array(batch * seqLen * inputSize);
    this.gradWxh = new Tensor(new Float32Array(this.Wxh.data.length), this.Wxh.shape);
    this.gradWhh = new Tensor(new Float32Array(this.Whh.data.length), this.Whh.shape);
    this.gradBh = new Tensor(new Float32Array(hiddenSize), [hiddenSize]);
    let dhNext = new Tensor(new Float32Array(batch * hiddenSize).fill(0), [batch, hiddenSize]);
    for (let t = seqLen - 1; t >= 0; t--) {
      const dh = grad.slice(t * batch * hiddenSize, (t + 1) * batch * hiddenSize).add(dhNext);
      // dh_raw = dh * (1 - h^2)
      const h = this.hs[t + 1];
      const dhRaw = dh.mul(h.mul(h).mul(-1).add(1));
      this.gradBh = this.gradBh!.add(dhRaw.sum(0)); // sum over batch
      this.gradWxh = this.gradWxh!.add(this.inputs[t].transpose().matmul(dhRaw));
      this.gradWhh = this.gradWhh!.add(this.hs[t].transpose().matmul(dhRaw));
      dhNext = dhRaw.matmul(this.Whh.transpose());
      // grad_x = dhRaw @ Wxh.T
      const gradX = dhRaw.matmul(this.Wxh.transpose());
      for (let i = 0; i < gradX.data.length; i++) {
        gradIn[t * batch * inputSize + i] = gradX.data[i];
      }
    }
    return new Tensor(gradIn, [batch, seqLen, inputSize]);
  }

  params(): Tensor[] {
    return [this.Wxh, this.Whh, this.bh];
  }

  grads(): Tensor[] {
    return [this.gradWxh!, this.gradWhh!, this.gradBh!];
  }
}

export class RNNModel {
  static build(config: RNNConfig): Layer[] {
    return [new RNN(config.inputSize, config.hiddenSize), new Dense(config.hiddenSize, config.outputSize)];
  }
}