import { Tensor } from "./tensor";
import { Optimizer } from "./optimizer";
import { Loss } from "./loss";
import { LRScheduler } from "./scheduler";

export interface Layer {
  type: string;
  config: any;
  training?: boolean;
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

  setTraining(mode: boolean): void {
    for (const layer of this.layers) {
      layer.training = mode;
    }
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
  scheduler?: LRScheduler;

  constructor(config: { optimizer: Optimizer; loss: Loss; epochs: number; scheduler?: LRScheduler }) {
    this.optimizer = config.optimizer;
    this.lossFn = config.loss;
    this.epochs = config.epochs;
    this.scheduler = config.scheduler;
  }

  fit(model: Model, inputs: Tensor[], targets: Tensor[], validation?: { inputs: Tensor[], targets: Tensor[] }, onEpochEnd?: (epoch: number, loss: number, valLoss?: number, model?: Model) => void): void {
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      if (this.scheduler) {
        this.scheduler.step(epoch);
      }
      model.setTraining(true);
      let totalLoss = 0;
      for (let i = 0; i < inputs.length; i++) {
        const pred = model.forward(inputs[i]);
        const loss = this.lossFn.forward(pred, targets[i]);
        totalLoss += loss;
        const grad = this.lossFn.backward(pred, targets[i]);
        model.backward(grad);
        this.optimizer.step(model.params(), model.grads());
      }
      
      const avgLoss = totalLoss / inputs.length;
      let valLoss: number | undefined;

      if (validation) {
        valLoss = this.evaluate(model, validation.inputs, validation.targets);
      }

      console.log(`Epoch ${epoch + 1}/${this.epochs}, Loss: ${avgLoss.toFixed(6)}${valLoss !== undefined ? `, Val Loss: ${valLoss.toFixed(6)}` : ""}${this.scheduler ? `, LR: ${this.optimizer.lr.toFixed(8)}` : ""}`);
      
      if (onEpochEnd) {
        onEpochEnd(epoch + 1, avgLoss, valLoss, model);
      }
    }
  }

  evaluate(model: Model, inputs: Tensor[], targets: Tensor[]): number {
    model.setTraining(false);
    let totalLoss = 0;
    for (let i = 0; i < inputs.length; i++) {
      const pred = model.forward(inputs[i]);
      totalLoss += this.lossFn.forward(pred, targets[i]);
    }
    return totalLoss / inputs.length;
  }
}

export class Checkpoint {
  private bestLoss = Infinity;
  private path: string;
  private saveFn: (path: string, model: Model) => Promise<void>;

  constructor(path: string, saveFn: (path: string, model: Model) => Promise<void>) {
    this.path = path;
    this.saveFn = saveFn;
  }

  async onEpochEnd(epoch: number, loss: number, valLoss?: number, model?: Model): Promise<void> {
    const currentLoss = valLoss !== undefined ? valLoss : loss;
    if (currentLoss < this.bestLoss && model) {
      this.bestLoss = currentLoss;
      await this.saveFn(this.path, model);
      console.log(`Checkpoint saved: best loss ${this.bestLoss.toFixed(6)}`);
    }
  }
}