import { describe, it, expect } from "vitest";
import { matmul, add, transpose } from "../../src/core/math";
import { Tensor } from "../../src/core/tensor";

describe('Math', () => {
  it('should matmul', () => {
    const a = new Tensor([1,2,3,4], [2,2]);
    const b = new Tensor([5,6,7,8], [2,2]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2,2]);
    expect(c.data).toEqual(new Float32Array([19,22,43,50]));
  });

  it('should add tensors', () => {
    const a = new Tensor([1,2], [2]);
    const b = new Tensor([3,4], [2]);
    const c = add(a, b);
    expect(c.data).toEqual(new Float32Array([4,6]));
  });

  it('should add scalar', () => {
    const a = new Tensor([1,2], [2]);
    const c = add(a, 10);
    expect(c.data).toEqual(new Float32Array([11,12]));
  });

  it('should transpose', () => {
    const a = new Tensor([1,2,3,4], [2,2]);
    const b = transpose(a);
    expect(b.shape).toEqual([2,2]);
    expect(b.data).toEqual(new Float32Array([1,3,2,4]));
  });
});