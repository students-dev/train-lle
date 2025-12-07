import { Model } from "../../src/core/engine";
import { Trainer } from "../../src/core/engine";
import { MLP } from "../../src/models/mlp";
import { SGD } from "../../src/core/optimizer";
import { MSE } from "../../src/core/loss";
import { Dataset } from "../../src/dataset/csv";
import { Tensor } from "../../src/core/tensor";

// Generate synthetic data
const numSamples = 100;
const inputs = new Float32Array(numSamples * 4);
const targets = new Float32Array(numSamples);
for (let i = 0; i < numSamples; i++) {
  for (let j = 0; j < 4; j++) {
    inputs[i * 4 + j] = Math.random();
  }
  targets[i] = inputs[i * 4] + inputs[i * 4 + 1] * 2 + inputs[i * 4 + 2] * 3 + inputs[i * 4 + 3] * 4;
}
const dataset = new Dataset(new Tensor(inputs, [numSamples, 4]), new Tensor(targets, [numSamples, 1]));
const model = new Model(MLP.build({ input: 4, layers: [8, 8], output: 1 }));
const trainer = new Trainer({ optimizer: new SGD(0.01), loss: new MSE(), epochs: 100 });
trainer.fit(model, dataset.inputs, dataset.targets);
console.log('Training complete');