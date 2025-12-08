# Train-LLE Ecosystem v1.1.2

**Local Learning Engine** - A document-first LOCAL LEARNING ENGINE ecosystem with TypeScript, JavaScript, and Python packages. Train neural networks from PDFs, DOCX, images, code, and more, all locally without cloud dependencies.

## Features

- **Document-first**: Ingest and train from PDFs, DOCX, HTML, TXT, images, code, emails, ZIPs
- **Local-first**: No internet required for training or inference
- **Cross-language**: TypeScript, JavaScript, Python with identical behavior
- **Simple API**: Easy to use for tabular, image, text, and document data
- **Models**: MLP, CNN, RNN implementations
- **Optimizers**: SGD, Adam, RMSProp
- **Loss functions**: MSE, MAE, CrossEntropy
- **Activations**: ReLU, Sigmoid, Tanh, Softmax
- **CLI**: Command-line interface for ingestion, training, and export workflows
- **Cross-language format**: Save/load models in `.lle` v1.1 format
- **Dataset manifest**: Structured dataset handling with splits and provenance

## Installation

### TypeScript (ESM)
```bash
npm install @students-dev/train-lle
# or
pnpm add @students-dev/train-lle
```

### JavaScript (CommonJS)
```bash
npm install @students-dev/train-lle.js
```

### Python
```bash
pip install train-lle
```

## Quickstart

### Programmatic Usage

```ts
import { Model, MLPConfig, Trainer, Dataset } from "@students-dev/train-lle";

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
# Ingest documents
npx train-lle ingest /path/to/documents

# Extract text
npx train-lle extract artifacts.json

# Assemble dataset
npx train-lle assemble-dataset manifest.json

# Index for retrieval
npx train-lle index dataset/

# Train from corpus
npx train-lle train-from-corpus dataset/

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

- `ingest`: Ingest files from path
- `extract`: Extract text from artifacts
- `assemble-dataset`: Assemble dataset from extracted artifacts
- `index`: Index dataset for retrieval
- `train-from-corpus`: Train from document corpus
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