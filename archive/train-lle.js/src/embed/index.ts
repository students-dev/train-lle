import { HierarchicalNSW } from 'hnswlib-node';

export class EmbeddingModel {
  private vocab: Map<string, number> = new Map();
  private vectors: number[][] = [];

  embed(text: string): number[] {
    const tokens = text.toLowerCase().split(/\s+/);
    const vector = new Array(128).fill(0);
    for (const token of tokens) {
      let index = this.vocab.get(token);
      if (index === undefined) {
        index = this.vocab.size;
        this.vocab.set(token, index);
        // Random vector
        this.vectors[index] = Array.from({ length: 128 }, () => Math.random() - 0.5);
      }
      for (let i = 0; i < 128; i++) {
        vector[i] += this.vectors[index][i];
      }
    }
    // Normalize
    const norm = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
    return vector.map(v => v / norm);
  }
}

export class VectorIndex {
  private index: HierarchicalNSW;
  private vectors: number[][] = [];

  constructor(dim: number = 128) {
    this.index = new HierarchicalNSW('cosine', dim);
    this.index.initIndex(1000); // max elements
  }

  add(vector: number[]): number {
    const id = this.vectors.length;
    this.vectors.push(vector);
    this.index.addPoint(vector, id);
    return id;
  }

  search(query: number[], k: number = 10): number[] {
    return this.index.searchKnn(query, k).neighbors;
  }
}