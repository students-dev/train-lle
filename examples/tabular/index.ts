import { Model } from "../../packages/train-lle/src/core/engine";
import { Trainer } from "../../packages/train-lle/src/core/engine";
import { MLP } from "../../packages/train-lle/src/models/mlp";
import { SGD } from "../../packages/train-lle/src/core/optimizer";
import { MSE } from "../../packages/train-lle/src/core/loss";
import { Dataset } from "../../packages/train-lle/src/dataset/csv";
import { Tensor } from "../../packages/train-lle/src/core/tensor";

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