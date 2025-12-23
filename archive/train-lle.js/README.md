# train-lle.js

JavaScript CommonJS package for the Local Learning Engine.

## Installation

```bash
npm install @students-dev/train-lle.js
```

## Usage

```js
const { Model, MLP, Trainer, Dataset } = require("@students-dev/train-lle.js");

const model = new Model(MLP.build({ input: 4, layers: [8, 8], output: 1 }));
const trainer = new Trainer({ optimizer: "adam", lr: 0.01, epochs: 50, loss: "mse" });

const dataset = Dataset.fromCSV("data.csv");
trainer.fit(model, dataset.inputs, dataset.targets);

model.save("model.lle");
```