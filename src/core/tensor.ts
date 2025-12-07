/**
 * Minimal tensor class wrapping Float32Array for multi-dimensional arrays.
 */
export class Tensor {
  data: Float32Array;
  shape: number[];

  constructor(data: number[] | Float32Array, shape?: number[]) {
    this.data = data instanceof Float32Array ? data : new Float32Array(data);
    if (shape) {
      this.shape = shape;
    } else {
      this.shape = [this.data.length];
    }
  }

  /**
   * Reshape the tensor to new shape.
   */
  reshape(newShape: number[]): Tensor {
    const total = newShape.reduce((a, b) => a * b, 1);
    if (total !== this.data.length) {
      throw new Error("Cannot reshape: total elements mismatch");
    }
    this.shape = newShape;
    return this;
  }

  /**
   * Get element at index (flat).
   */
  get(index: number): number {
    return this.data[index];
  }

  /**
   * Set element at index.
   */
  set(index: number, value: number): void {
    this.data[index] = value;
  }

  /**
   * Element-wise addition.
   */
  add(other: Tensor | number): Tensor {
    const result = new Float32Array(this.data.length);
    if (typeof other === "number") {
      for (let i = 0; i < this.data.length; i++) {
        result[i] = this.data[i] + other;
      }
    } else {
      for (let i = 0; i < this.data.length; i++) {
        result[i] = this.data[i] + other.data[i];
      }
    }
    return new Tensor(result, this.shape);
  }

  /**
   * Element-wise multiplication.
   */
  mul(other: Tensor | number): Tensor {
    const result = new Float32Array(this.data.length);
    if (typeof other === "number") {
      for (let i = 0; i < this.data.length; i++) {
        result[i] = this.data[i] * other;
      }
    } else {
      for (let i = 0; i < this.data.length; i++) {
        result[i] = this.data[i] * other.data[i];
      }
    }
    return new Tensor(result, this.shape);
  }

  /**
   * Matrix multiplication (simple 2D).
   */
  matmul(other: Tensor): Tensor {
    if (this.shape.length !== 2 || other.shape.length !== 2) {
      throw new Error("Matmul requires 2D tensors");
    }
    const [m, k] = this.shape;
    const [k2, n] = other.shape;
    if (k !== k2) {
      throw new Error("Incompatible shapes for matmul");
    }
    const result = new Float32Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let l = 0; l < k; l++) {
          sum += this.data[i * k + l] * other.data[l * n + j];
        }
        result[i * n + j] = sum;
      }
    }
    return new Tensor(result, [m, n]);
  }

  /**
   * Transpose (simple 2D).
   */
  transpose(): Tensor {
    if (this.shape.length !== 2) {
      throw new Error("Transpose requires 2D tensor");
    }
    const [m, n] = this.shape;
    const result = new Float32Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        result[j * m + i] = this.data[i * n + j];
      }
    }
    return new Tensor(result, [n, m]);
  }

  /**
   * Element-wise square root.
   */
  sqrt(): Tensor {
    const result = new Float32Array(this.data.length);
    for (let i = 0; i < this.data.length; i++) {
      result[i] = Math.sqrt(this.data[i]);
    }
    return new Tensor(result, this.shape);
  }

  /**
   * Element-wise tanh.
   */
  tanh(): Tensor {
    const result = new Float32Array(this.data.length);
    for (let i = 0; i < this.data.length; i++) {
      result[i] = Math.tanh(this.data[i]);
    }
    return new Tensor(result, this.shape);
  }

  /**
   * Sum over axis or all.
   */
  sum(axis?: number): Tensor {
    if (axis === 0 && this.shape.length === 2) {
      const [dim0, dim1] = this.shape;
      const out = new Float32Array(dim1);
      for (let j = 0; j < dim1; j++) {
        let s = 0;
        for (let i = 0; i < dim0; i++) {
          s += this.data[i * dim1 + j];
        }
        out[j] = s;
      }
      return new Tensor(out, [dim1]);
    }
    let s = 0;
    for (const v of this.data) s += v;
    return new Tensor([s], [1]);
  }

  /**
   * Slice data from start to end.
   */
  slice(start: number, end: number): Tensor {
    const sliced = this.data.slice(start, end);
    return new Tensor(sliced, [sliced.length]);
  }

  /**
   * Clone the tensor.
   */
  clone(): Tensor {
    return new Tensor(new Float32Array(this.data), [...this.shape]);
  }
}