import { Tensor } from "../core/tensor";
import { Dataset } from "./csv";

export interface DataLoaderConfig {
  batchSize: number;
  shuffle?: boolean;
  dropLast?: boolean;
}

export class DataLoader {
  dataset: Dataset;
  batchSize: number;
  shuffle: boolean;
  dropLast: boolean;
  indices: number[];

  constructor(dataset: Dataset, config: DataLoaderConfig) {
    this.dataset = dataset;
    this.batchSize = config.batchSize;
    this.shuffle = config.shuffle || false;
    this.dropLast = config.dropLast || false;
    this.indices = Array.from({ length: dataset.inputs.length }, (_, i) => i);
    this.reset();
  }

  reset(): void {
    if (this.shuffle) {
      // Fisher-Yates shuffle
      for (let i = this.indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [this.indices[i], this.indices[j]] = [this.indices[j], this.indices[i]];
      }
    }
  }

  *[Symbol.iterator](): Iterator<{ inputs: Tensor; targets: Tensor }> {
    this.reset();
    const len = this.dataset.inputs.length;
    
    // Check dataset consistency
    if (len !== this.dataset.targets.length) {
        throw new Error("Dataset inputs and targets must have same length");
    }

    const inputShape = this.dataset.inputs[0].shape;
    const targetShape = this.dataset.targets[0].shape;

    // Helper to slice batch from array of Tensors is tricky if they are just an array of Tensors.
    // The Dataset class currently stores inputs: Tensor[] | Tensor.
    // Let's assume standardized access or handle both.
    // Actually Dataset in csv.ts has `inputs: Tensor[]` and `targets: Tensor[]` (based on usage in trainer).
    // Wait, let's check csv.ts.

    for (let i = 0; i < len; i += this.batchSize) {
      if (this.dropLast && i + this.batchSize > len) {
        break;
      }
      
      const end = Math.min(i + this.batchSize, len);
      const batchIndices = this.indices.slice(i, end);
      
      // Collect batch data
      // Optimization: If dataset stores as big Tensor, we could slice. 
      // If array of Tensors, we stack.
      // Based on previous reads, Dataset seems to have properties inputs/targets.
      // Let's implement generic stacking.
      
      // Efficiently stack
      const batchSize = batchIndices.length;
      
      // Flatten data for the batch
      // input size = batchSize * input_dim
      const inputTotalSize = batchSize * this.dataset.inputs[0].data.length;
      const targetTotalSize = batchSize * this.dataset.targets[0].data.length;
      
      const inputData = new Float32Array(inputTotalSize);
      const targetData = new Float32Array(targetTotalSize);
      
      const oneInputLen = this.dataset.inputs[0].data.length;
      const oneTargetLen = this.dataset.targets[0].data.length;

      for (let b = 0; b < batchSize; b++) {
        const idx = batchIndices[b];
        inputData.set(this.dataset.inputs[idx].data, b * oneInputLen);
        targetData.set(this.dataset.targets[idx].data, b * oneTargetLen);
      }
      
      const batchInputs = new Tensor(inputData, [batchSize, ...inputShape]);
      const batchTargets = new Tensor(targetData, [batchSize, ...targetShape]);
      
      yield { inputs: batchInputs, targets: batchTargets };
    }
  }

  get length(): number {
    if (this.dropLast) {
      return Math.floor(this.dataset.inputs.length / this.batchSize);
    }
    return Math.ceil(this.dataset.inputs.length / this.batchSize);
  }
}
