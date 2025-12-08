# train-lle

TypeScript ESM package for the Local Learning Engine.

## Installation

```bash
npm install train-lle
# or
pnpm add train-lle
```

## Usage

```ts
import { Model, MLP, Trainer, Dataset } from "train-lle";

const model = new Model(MLP.build({ input: 4, layers: [8, 8], output: 1 }));
const trainer = new Trainer({ optimizer: "adam", lr: 0.01, epochs: 50, loss: "mse" });

const dataset = Dataset.fromCSV("data.csv");
trainer.fit(model, dataset.inputs, dataset.targets);

model.save("model.lle");
```