import { writeFileSync, readFileSync } from "fs";
import { Model } from "../core/engine";
import { Tensor } from "../core/tensor";

export function saveModel(path: string, model: Model): void {
  const params = model.params();
  const data = {
    shapes: params.map(p => p.shape),
    weights: params.map(p => Buffer.from(p.data.buffer).toString('base64'))
  };
  writeFileSync(path, JSON.stringify(data));
}

export function loadModel(path: string): Model {
  const data = JSON.parse(readFileSync(path, 'utf-8'));
  const params: Tensor[] = [];
  for (let i = 0; i < data.shapes.length; i++) {
    const buf = Buffer.from(data.weights[i], 'base64');
    const arr = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
    params.push(new Tensor(arr, data.shapes[i]));
  }
  // For MVP, assume MLP, and reconstruct layers.
  // This is incomplete, but placeholder.
  throw new Error('Load not fully implemented');
}