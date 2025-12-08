import { readFileSync } from "fs";
import { Tensor } from "../core/tensor";

export class Dataset {
  inputs: Tensor[];
  targets: Tensor[];

  constructor(inputs: Tensor[], targets: Tensor[]) {
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
    const numCols = rows[0].length;
    const inputs: Tensor[] = [];
    const targets: Tensor[] = [];
    for (const row of rows) {
      const inputData = row.slice(0, numCols - 1);
      const targetData = [row[numCols - 1]];
      inputs.push(new Tensor(inputData, [numCols - 1]));
      targets.push(new Tensor(targetData, [1]));
    }
    return new Dataset(inputs, targets);
  }
}