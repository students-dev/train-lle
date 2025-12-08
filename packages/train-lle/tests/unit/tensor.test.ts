import { describe, it, expect } from "vitest";
import { Tensor } from "../../src/core/tensor";

describe('Tensor', () => {
  it('should create tensor', () => {
    const t = new Tensor([1, 2, 3], [3]);
    expect(t.shape).toEqual([3]);
    expect(t.data).toEqual(new Float32Array([1,2,3]));
  });

  it('should add', () => {
    const a = new Tensor([1,2], [2]);
    const b = new Tensor([3,4], [2]);
    const c = a.add(b);
    expect(c.data).toEqual(new Float32Array([4,6]));
  });

  it('should matmul', () => {
    const a = new Tensor([1,2,3,4], [2,2]);
    const b = new Tensor([5,6,7,8], [2,2]);
    const c = a.matmul(b);
    expect(c.shape).toEqual([2,2]);
  });

  it('should tanh', () => {
    const a = new Tensor([0], [1]);
    const b = a.tanh();
    expect(b.data[0]).toBeCloseTo(0);
  });

  it('should sum', () => {
    const a = new Tensor([1,2,3,4], [2,2]);
    const b = a.sum(0);
    expect(b.data).toEqual(new Float32Array([4,6]));
  });

  it('should slice', () => {
    const a = new Tensor([1,2,3,4], [4]);
    const b = a.slice(1,3);
    expect(b.data).toEqual(new Float32Array([2,3]));
  });
});