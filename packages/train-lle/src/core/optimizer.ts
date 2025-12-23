import { Tensor } from "./tensor";

export interface Optimizer {
  lr: number;
  step(params: Tensor[], grads: Tensor[]): void;
}

/**
 * Stochastic Gradient Descent optimizer.
 */
export class SGD implements Optimizer {
  lr: number;

  constructor(lr: number = 0.01) {
    this.lr = lr;
  }

  step(params: Tensor[], grads: Tensor[]): void {
    for (let i = 0; i < params.length; i++) {
      params[i].data = params[i].add(grads[i].mul(-this.lr)).data;
    }
  }
}

/**
 * Adam optimizer.
 */
export class Adam implements Optimizer {
  lr: number;
  beta1: number;
  beta2: number;
  epsilon: number;
  t: number;
  m: Tensor[];
  v: Tensor[];

  constructor(lr: number = 0.001, beta1: number = 0.9, beta2: number = 0.999, epsilon: number = 1e-8) {
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
    this.t = 0;
    this.m = [];
    this.v = [];
  }

  step(params: Tensor[], grads: Tensor[]): void {
    this.t++;
    if (this.m.length === 0) {
      this.m = grads.map(g => new Tensor(new Float32Array(g.data.length), g.shape));
      this.v = grads.map(g => new Tensor(new Float32Array(g.data.length), g.shape));
    }
    for (let i = 0; i < params.length; i++) {
      // m = beta1 * m + (1 - beta1) * grad
      this.m[i] = this.m[i].mul(this.beta1).add(grads[i].mul(1 - this.beta1));
      // v = beta2 * v + (1 - beta2) * grad^2
      this.v[i] = this.v[i].mul(this.beta2).add(grads[i].mul(grads[i]).mul(1 - this.beta2));
      // m_hat = m / (1 - beta1^t)
      const mHat = this.m[i].mul(1 / (1 - Math.pow(this.beta1, this.t)));
      // v_hat = v / (1 - beta2^t)
      const vHat = this.v[i].mul(1 / (1 - Math.pow(this.beta2, this.t)));
      // params -= lr * m_hat / (sqrt(v_hat) + epsilon)
      const denom = vHat.sqrt().add(this.epsilon);
      const update = mHat.mul(this.lr).mul(denom.mul(-1));
      params[i].data = params[i].add(update).data;
    }
  }
}

/**
 * RMSProp optimizer.
 */
export class RMSProp implements Optimizer {
  lr: number;
  beta: number;
  epsilon: number;
  v: Tensor[];

  constructor(lr: number = 0.001, beta: number = 0.9, epsilon: number = 1e-8) {
    this.lr = lr;
    this.beta = beta;
    this.epsilon = epsilon;
    this.v = [];
  }

  step(params: Tensor[], grads: Tensor[]): void {
    if (this.v.length === 0) {
      this.v = grads.map(g => new Tensor(new Float32Array(g.data.length), g.shape));
    }
    for (let i = 0; i < params.length; i++) {
      // v = beta * v + (1 - beta) * grad^2
      this.v[i] = this.v[i].mul(this.beta).add(grads[i].mul(grads[i]).mul(1 - this.beta));
      // params -= lr * grad / (sqrt(v) + epsilon)
      const denom = this.v[i].sqrt().add(this.epsilon);
      const update = grads[i].mul(this.lr).mul(denom.mul(-1));
      params[i].data = params[i].add(update).data;
    }
  }
}