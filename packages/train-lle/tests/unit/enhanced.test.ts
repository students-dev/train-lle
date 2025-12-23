import { describe, it, expect } from "vitest";
import { DataLoader } from "../../src/dataset/dataloader";
import { Dataset } from "../../src/dataset/csv";
import { Tensor } from "../../src/core/tensor";
import { Metrics } from "../../src/core/metrics";
import { AdamW } from "../../src/core/optimizer";

describe('Enhanced Features', () => {
  describe('DataLoader', () => {
    it('should iterate over batches', () => {
      // 10 samples, batch size 3 -> 3, 3, 3, 1
      const inputs = Array.from({ length: 10 }, (_, i) => new Tensor([i], [1]));
      const targets = Array.from({ length: 10 }, (_, i) => new Tensor([i], [1]));
      // Mock Dataset
      const dataset = { inputs, targets } as any as Dataset;
      
      const loader = new DataLoader(dataset, { batchSize: 3, shuffle: false });
      const batches = [];
      for (const batch of loader) {
        batches.push(batch);
      }
      
      expect(batches.length).toBe(4);
      expect(batches[0].inputs.shape[0]).toBe(3);
      expect(batches[3].inputs.shape[0]).toBe(1);
    });

    it('should shuffle', () => {
      const inputs = Array.from({ length: 10 }, (_, i) => new Tensor([i], [1]));
      const targets = Array.from({ length: 10 }, (_, i) => new Tensor([i], [1]));
      const dataset = { inputs, targets } as any as Dataset;
      
      const loader = new DataLoader(dataset, { batchSize: 10, shuffle: true });
      let firstBatchIndices: number[] = [];
      for (const batch of loader) {
        firstBatchIndices = Array.from(batch.inputs.data);
      }
      
      // Check if order is different from 0..9
      // Small chance it is same, but negligible
      let isSame = true;
      for (let i = 0; i < 10; i++) {
        if (firstBatchIndices[i] !== i) isSame = false;
      }
      expect(isSame).toBe(false);
    });
  });

  describe('Metrics', () => {
    it('should calculate accuracy', () => {
        const preds = new Tensor([0.1, 0.9, 0.8, 0.2], [2, 2]); // Class 1, Class 0
        const targets = new Tensor([0, 1, 1, 0], [2, 2]); // Class 1, Class 0
        const acc = Metrics.accuracy(preds, targets);
        expect(acc).toBe(1.0);
    });

    it('should calculate MSE', () => {
        const preds = new Tensor([1, 2], [2]);
        const targets = new Tensor([2, 3], [2]);
        // (1-2)^2 + (2-3)^2 = 1 + 1 = 2 / 2 = 1
        expect(Metrics.mse(preds, targets)).toBe(1.0);
    });
  });

  describe('AdamW', () => {
    it('should update params with weight decay', () => {
        const param = new Tensor([1.0], [1]);
        const grad = new Tensor([0.0], [1]); // Zero grad, only weight decay should act
        const opt = new AdamW(0.1, 0.9, 0.999, 1e-8, 0.1); // lr=0.1, wd=0.1
        
        opt.step([param], [grad]);
        // param = param * (1 - lr * wd) = 1.0 * (1 - 0.1*0.1) = 1.0 * 0.99 = 0.99
        // plus adam update (which is 0 since grad is 0)
        expect(param.data[0]).toBeCloseTo(0.99);
    });
  });
});
