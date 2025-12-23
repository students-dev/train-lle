import { describe, it, expect } from "vitest";
import { Tensor } from "../../src/core/tensor";
import { Dropout, BatchNormalization } from "../../src/models/mlp";

describe('Layers', () => {
  describe('Dropout', () => {
    it('should pass input as-is during evaluation', () => {
      const dropout = new Dropout(0.5);
      dropout.training = false;
      const input = new Tensor([1, 2, 3, 4], [4]);
      const output = dropout.forward(input);
      expect(output.data).toEqual(input.data);
    });

    it('should apply mask during training', () => {
      const dropout = new Dropout(0.5);
      dropout.training = true;
      const input = new Tensor(new Float32Array(100).fill(1), [100]);
      const output = dropout.forward(input);
      
      let zeros = 0;
      for (let i = 0; i < output.data.length; i++) {
        if (output.data[i] === 0) zeros++;
      }
      
      // With 100 elements and 0.5 rate, we expect roughly 50 zeros.
      // We check that at least some elements are zero and some are not.
      expect(zeros).toBeGreaterThan(0);
      expect(zeros).toBeLessThan(100);
    });
  });

  describe('BatchNormalization', () => {
    it('should normalize batch to mean 0 and var 1', () => {
      const bn = new BatchNormalization(2);
      bn.training = true;
      // 2 features, 3 samples
      const input = new Tensor([
        1, 10,
        2, 20,
        3, 30
      ], [3, 2]);
      
      const output = bn.forward(input);
      
      // Feature 1: mean 2, std sqrt(2/3 * sum((x-2)^2)) = sqrt(2/3 * (1+0+1)) = sqrt(2/3)
      // Actually my implementation uses biased variance: sum((x-mean)^2)/N
      // mean = (1+2+3)/3 = 2
      // var = ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = (1+0+1)/3 = 2/3
      // x_hat = (x - mean) / sqrt(var + eps)
      
      // Feature 1 normalized should have mean ~0 and var ~1
      let sum = 0;
      for(let i=0; i<3; i++) sum += output.data[i*2];
      expect(sum/3).toBeCloseTo(0, 5);
      
      let sumSq = 0;
      for(let i=0; i<3; i++) sumSq += Math.pow(output.data[i*2], 2);
      expect(sumSq/3).toBeCloseTo(1, 1);
    });
  });
});
