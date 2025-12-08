import { Tensor } from "./tensor";
import { Optimizer } from "./optimizer";
import { Loss } from "./loss";

export interface Layer {
  type: string;
  config: any;
  forward(input: Tensor): Tensor;
  backward(grad: Tensor): Tensor;
  params(): Tensor[];
  grads(): Tensor[];
}

export class Model {
  layers: Layer[];

  constructor(layers: Layer[]) {
    this.layers = layers;
  }

  forward(input: Tensor): Tensor {
    let out = input;
    for (const layer of this.layers) {
      out = layer.forward(out);
    }
    return out;
  }

  backward(grad: Tensor): void {
    let g = grad;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      g = this.layers[i].backward(g);
    }
  }

  params(): Tensor[] {
    return this.layers.flatMap(l => l.params());
  }

  grads(): Tensor[] {
    return this.layers.flatMap(l => l.grads());
  }

  predict(input: Tensor): Tensor {
    return this.forward(input);
  }
}

export class Trainer {
  optimizer: Optimizer;
  lossFn: Loss;
  epochs: number;

  constructor(config: { optimizer: Optimizer; loss: Loss; epochs: number }) {
    this.optimizer = config.optimizer;
    this.lossFn = config.loss;
    this.epochs = config.epochs;
  }

  fit(model: Model, inputs: Tensor, targets: Tensor): void {
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      const pred = model.forward(inputs);
      const loss = this.lossFn.forward(pred, targets);
      const grad = this.lossFn.backward(pred, targets);
      model.backward(grad);
      this.optimizer.step(model.params(), model.grads());
      console.log(`Epoch ${epoch + 1}/${this.epochs}, Loss: ${loss}`);
    }
  }

  evaluate(model: Model, inputs: Tensor, targets: Tensor): number {
    const pred = model.forward(inputs);
    return this.lossFn.forward(pred, targets);
  }
}