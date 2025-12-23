import { Tensor } from "../core/tensor";
import { Layer } from "../core/engine";

export class Conv2D implements Layer {
  type = "conv2d";
  config: { inChannels: number; outChannels: number; kernelSize: number };
  weight: Tensor; // [outChannels, inChannels, kH, kW]
  bias: Tensor; // [outChannels]
  gradW: Tensor | null = null;
  gradB: Tensor | null = null;
  input: Tensor | null = null;
  output: Tensor | null = null;
  stride: number = 1;

  constructor(inChannels: number, outChannels: number, kernelSize: number) {
    this.config = { inChannels, outChannels, kernelSize };
    this.weight = new Tensor(new Float32Array(outChannels * inChannels * kernelSize * kernelSize).map(() => Math.random() - 0.5), [outChannels, inChannels, kernelSize, kernelSize]);
    this.bias = new Tensor(new Float32Array(outChannels).fill(0), [outChannels]);
  }

  forward(input: Tensor): Tensor {
    // Assume input [batch, inC, H, W]
    this.input = input;
    const [batch, inC, H, W] = input.shape;
    const [outC, , kH, kW] = this.weight.shape;
    const outH = H - kH + 1; // no padding
    const outW = W - kW + 1;
    const out = new Float32Array(batch * outC * outH * outW);
    for (let b = 0; b < batch; b++) {
      for (let oc = 0; oc < outC; oc++) {
        for (let oh = 0; oh < outH; oh++) {
          for (let ow = 0; ow < outW; ow++) {
            let sum = this.bias.data[oc];
            for (let ic = 0; ic < inC; ic++) {
              for (let kh = 0; kh < kH; kh++) {
                for (let kw = 0; kw < kW; kw++) {
                  const ih = oh + kh;
                  const iw = ow + kw;
                  const inVal = input.data[b * inC * H * W + ic * H * W + ih * W + iw];
                  const wVal = this.weight.data[oc * inC * kH * kW + ic * kH * kW + kh * kW + kw];
                  sum += inVal * wVal;
                }
              }
            }
            out[b * outC * outH * outW + oc * outH * outW + oh * outW + ow] = sum;
          }
        }
      }
    }
    this.output = new Tensor(out, [batch, outC, outH, outW]);
    return this.output;
  }

  backward(grad: Tensor): Tensor {
    // Simplified backprop for conv
    // For MVP, implement basic grad computation
    const [batch, outC, outH, outW] = grad.shape;
    const [inB, inC, inH, inW] = this.input!.shape;
    const [, , kH, kW] = this.weight.shape;
    // grad_input
    const gradIn = new Float32Array(batch * inC * inH * inW);
    // grad_w
    this.gradW = new Tensor(new Float32Array(this.weight.data.length), this.weight.shape);
    this.gradB = new Tensor(new Float32Array(outC), [outC]);
    // For simplicity, accumulate grads
    for (let b = 0; b < batch; b++) {
      for (let oc = 0; oc < outC; oc++) {
        for (let oh = 0; oh < outH; oh++) {
          for (let ow = 0; ow < outW; ow++) {
            const g = grad.data[b * outC * outH * outW + oc * outH * outW + oh * outW + ow];
            this.gradB.data[oc] += g;
            for (let ic = 0; ic < inC; ic++) {
              for (let kh = 0; kh < kH; kh++) {
                for (let kw = 0; kw < kW; kw++) {
                  const ih = oh + kh;
                  const iw = ow + kw;
                  const inVal = this.input!.data[b * inC * inH * inW + ic * inH * inW + ih * inW + iw];
                  this.gradW.data[oc * inC * kH * kW + ic * kH * kW + kh * kW + kw] += g * inVal;
                  gradIn[b * inC * inH * inW + ic * inH * inW + ih * inW + iw] += g * this.weight.data[oc * inC * kH * kW + ic * kH * kW + kh * kW + kw];
                }
              }
            }
          }
        }
      }
    }
    return new Tensor(gradIn, this.input!.shape);
  }

  params(): Tensor[] {
    return [this.weight, this.bias];
  }

  grads(): Tensor[] {
    return [this.gradW!, this.gradB!];
  }
}

export class Flatten implements Layer {
  type = "flatten";
  config = {};
  inputShape: number[] | null = null;

  forward(input: Tensor): Tensor {
    this.inputShape = input.shape;
    if (input.shape.length > 1) {
       // Preserves batch dim 0
       const batch = input.shape[0];
       const featureSize = input.data.length / batch;
       return new Tensor(input.data, [batch, featureSize]);
    }
    return new Tensor(input.data, [input.data.length]);
  }

  backward(grad: Tensor): Tensor {
    return new Tensor(grad.data, this.inputShape!);
  }

  params(): Tensor[] {
    return [];
  }

  grads(): Tensor[] {
    return [];
  }
}

export interface CNNConfig {
  inputShape: [number, number, number]; // [C, H, W]
  conv: { outChannels: number; kernelSize: number };
  dense: number[];
  output: number;
}

export class CNN {
  static build(config: CNNConfig): Layer[] {
    const layers: Layer[] = [];
    layers.push(new Conv2D(config.inputShape[0], config.conv.outChannels, config.conv.kernelSize));
    layers.push(new ReLU()); // assume relu
    layers.push(new Flatten());
    let prev = config.conv.outChannels * (config.inputShape[1] - config.conv.kernelSize + 1) * (config.inputShape[2] - config.conv.kernelSize + 1);
    for (const d of config.dense) {
      layers.push(new Dense(prev, d));
      layers.push(new ReLU());
      prev = d;
    }
    layers.push(new Dense(prev, config.output));
    return layers;
  }
}

// Import ReLU from mlp
import { ReLU, Dense } from "./mlp";