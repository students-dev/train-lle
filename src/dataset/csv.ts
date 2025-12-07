import { readFileSync } from "fs";
import { Tensor } from "../core/tensor";

export class Dataset {
  inputs: Tensor;
  targets: Tensor;

  constructor(inputs: Tensor, targets: Tensor) {
    this.inputs = inputs;
    this.targets = targets;
  }

  static fromCSV(path: string): Dataset {
    const data = readFileSync(path, 'utf-8');
    const lines = data.split('\n').filter(l => l.trim());
    const rows: number[][] = [];
    for (const line of lines) {
      const cols = line.split(',').map(s => parseFloat(s.trim()));
      if (cols.some(isNaN)) continue;
      rows.push(cols);
    }
    const numRows = rows.length;
    const numCols = rows[0].length;
    const inputsData = new Float32Array(numRows * (numCols - 1));
    const targetsData = new Float32Array(numRows);
    for (let i = 0; i < numRows; i++) {
      for (let j = 0; j < numCols - 1; j++) {
        inputsData[i * (numCols - 1) + j] = rows[i][j];
      }
      targetsData[i] = rows[i][numCols - 1];
    }
    return new Dataset(new Tensor(inputsData, [numRows, numCols - 1]), new Tensor(targetsData, [numRows, 1]));
  }
}