import { readFileSync } from "fs";
import { Tensor } from "../core/tensor";
import { Dataset } from "./csv";

export class DatasetJSON {
  static load(path: string): Dataset {
    const data = JSON.parse(readFileSync(path, 'utf-8'));
    const numRows = data.length;
    const inputSize = data[0][0].length;
    const inputsData = new Float32Array(numRows * inputSize);
    const targetsData = new Float32Array(numRows);
    for (let i = 0; i < numRows; i++) {
      for (let j = 0; j < inputSize; j++) {
        inputsData[i * inputSize + j] = data[i][0][j];
      }
      targetsData[i] = data[i][1];
    }
    return new Dataset(new Tensor(inputsData, [numRows, inputSize]), new Tensor(targetsData, [numRows, 1]));
  }
}