import { Tensor } from "../../core/tensor";

/**
 * Naive matrix multiplication for 2D tensors.
 * @param a First matrix
 * @param b Second matrix
 * @returns Result of a * b
 */
export function matmul(a: Tensor, b: Tensor): Tensor {
  if (a.shape.length !== 2 || b.shape.length !== 2) {
    throw new Error("Matmul requires 2D tensors");
  }
  const [m, k] = a.shape;
  const [k2, n] = b.shape;
  if (k !== k2) {
    throw new Error("Incompatible shapes for matmul");
  }
  const result = new Float32Array(m * n);
  // Loop over rows of a
  for (let i = 0; i < m; i++) {
    // Loop over columns of b
    for (let j = 0; j < n; j++) {
      let sum = 0;
      // Loop over columns of a / rows of b
      for (let l = 0; l < k; l++) {
        sum += a.data[i * k + l] * b.data[l * n + j];
      }
      result[i * n + j] = sum;
    }
  }
  return new Tensor(result, [m, n]);
}

/**
 * Element-wise addition of two tensors or tensor and scalar.
 * @param a First tensor
 * @param b Second tensor or scalar
 * @returns Result tensor
 */
export function add(a: Tensor, b: Tensor | number): Tensor {
  const result = new Float32Array(a.data.length);
  if (typeof b === "number") {
    for (let i = 0; i < a.data.length; i++) {
      result[i] = a.data[i] + b;
    }
  } else {
    for (let i = 0; i < a.data.length; i++) {
      result[i] = a.data[i] + b.data[i];
    }
  }
  return new Tensor(result, a.shape);
}

/**
 * Transpose a 2D tensor.
 * @param a 2D tensor
 * @returns Transposed tensor
 */
export function transpose(a: Tensor): Tensor {
  if (a.shape.length !== 2) {
    throw new Error("Transpose requires 2D tensor");
  }
  const [m, n] = a.shape;
  const result = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      result[j * m + i] = a.data[i * n + j];
    }
  }
  return new Tensor(result, [n, m]);
}