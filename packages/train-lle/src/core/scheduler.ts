import { Optimizer } from "./optimizer";

export interface LRScheduler {
  step(epoch: number): void;
}

/**
 * Decays the learning rate of each parameter group by gamma every step_size epochs.
 */
export class StepLR implements LRScheduler {
  optimizer: Optimizer;
  stepSize: number;
  gamma: number;
  initialLR: number;

  constructor(optimizer: Optimizer, stepSize: number, gamma: number = 0.1) {
    this.optimizer = optimizer;
    this.stepSize = stepSize;
    this.gamma = gamma;
    this.initialLR = optimizer.lr;
  }

  step(epoch: number): void {
    const decay = Math.pow(this.gamma, Math.floor(epoch / this.stepSize));
    this.optimizer.lr = this.initialLR * decay;
  }
}

/**
 * Set the learning rate of each parameter group using a cosine annealing schedule.
 */
export class CosineAnnealing implements LRScheduler {
  optimizer: Optimizer;
  T_max: number;
  eta_min: number;
  initialLR: number;

  constructor(optimizer: Optimizer, T_max: number, eta_min: number = 0) {
    this.optimizer = optimizer;
    this.T_max = T_max;
    this.eta_min = eta_min;
    this.initialLR = optimizer.lr;
  }

  step(epoch: number): void {
    const lr = this.eta_min + (this.initialLR - this.eta_min) * (1 + Math.cos(Math.PI * epoch / this.T_max)) / 2;
    this.optimizer.lr = lr;
  }
}
