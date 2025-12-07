import { describe, it, expect } from "vitest";
import { MSE, CrossEntropy } from "../../src/core/loss";
import { Tensor } from "../../src/core/tensor";

describe('Loss', () => {
  it('should MSE', () => {
    const mse = new MSE();
    const pred = new Tensor([1,2], [2]);
    const target = new Tensor([1,2], [2]);
    const loss = mse.forward(pred, target);
    expect(loss).toBe(0);
  });

  it('should CrossEntropy', () => {
    const ce = new CrossEntropy();
    const pred = new Tensor([0.5,0.5], [2]);
    const target = new Tensor([1,0], [2]);
    const loss = ce.forward(pred, target);
    expect(loss).toBeGreaterThan(0);
  });
});