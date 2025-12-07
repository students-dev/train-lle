# train-lle

**Local Learning Engine** - A local-first ML training & inference engine for Node.js. Train neural networks on your machine without cloud dependencies.

## Features

- **Local-first**: No internet required for training or inference
- **TypeScript**: Fully typed for better developer experience
- **Simple API**: Easy to use for tabular, image, and text data
- **Models**: MLP, CNN, RNN implementations
- **Optimizers**: SGD and Adam
- **Loss functions**: MSE, CrossEntropy
- **CLI**: Command-line interface for training workflows
- **Custom format**: Save/load models in `.lle` format

## Installation

```bash
npm install train-lle
# or
pnpm add train-lle
```

## Quickstart

### Programmatic Usage

```ts
import { Model, MLPConfig, Trainer, Dataset } from "train-lle";

const model = new Model(new MLPConfig({ input: 4, layers: [8, 8], output: 1 }));
const trainer = new Trainer({ optimizer: "adam", lr: 0.01, epochs: 50 });

const dataset = await Dataset.fromCSV("data.csv");
await trainer.fit(model, dataset);

const prediction = model.predict([1, 2, 3, 4]);
console.log(prediction);

await model.save("model.lle");
```

### CLI Usage

```bash
# Initialize a project
npx train-lle init

# Train a model
npx train-lle train config.json

# Test on dataset
npx train-lle test model.lle dataset.csv

# Export model
npx train-lle export output.lle

# Show model stats
npx train-lle stats model.lle
```

## API Reference

### Model Classes

- `MLPConfig`: Configure multi-layer perceptron
- `CNNConfig`: Configure convolutional neural network
- `RNNConfig`: Configure recurrent neural network

### Core Classes

- `Model`: Neural network model
- `Trainer`: Training orchestrator
- `Dataset`: Data loading and preprocessing
- `Tensor`: Multi-dimensional array operations

### CLI Commands

- `init`: Create `train-config.json`
- `train`: Run training with config or dataset
- `test`: Evaluate model on dataset
- `export`: Save current model
- `stats`: Display model summary

## Examples

See `examples/` directory for runnable scripts:

- `examples/tabular/`: Regression on synthetic data
- `examples/image/`: Classification on tiny images
- `examples/text/`: Simple text classification

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run `npm test` and `npm run lint`
5. Submit a pull request

## License

Apache-2.0