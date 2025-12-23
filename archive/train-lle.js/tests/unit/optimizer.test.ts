import { describe, it, expect } from "vitest";
import { SGD, Adam } from "../../src/core/optimizer";
import { Tensor } from "../../src/core/tensor";

describe('Optimizer', () => {
  it('should SGD step', () => {
    const sgd = new SGD(0.1);
    const param = new Tensor([1,2], [2]);
    const grad = new Tensor([0.1,0.2], [2]);
    sgd.step([param], [grad]);
    expect(param.data).toEqual(new Float32Array([0.99,1.98]));
  });

  it('should Adam step', () => {
    const adam = new Adam(0.1);
    const param = new Tensor([1,2], [2]);
    const grad = new Tensor([0.1,0.2], [2]);
    adam.step([param], [grad]);
    // Approximate check
    expect(param.data[0]).toBeLessThan(1);
    expect(param.data[1]).toBeLessThan(2);
  });
});