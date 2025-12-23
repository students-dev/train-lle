# Train-LLE Ecosystem v1.2.0

**Local Learning Engine** - A document-first LOCAL LEARNING ENGINE ecosystem for Node.js. Train neural networks from PDFs, DOCX, images, code, and more, all locally without cloud dependencies.

## Features

- **Document-first**: Ingest and train from PDFs, DOCX, HTML, TXT, images, code, emails, ZIPs
- **Local-first**: No internet required for training or inference
- **Advanced Models**: **ResNet**, **Transformer**, MLP, CNN, RNN implementations
- **Optimizers**: SGD, Adam, RMSProp, **AdamW**
- **Schedulers**: **StepLR**, **CosineAnnealing**
- **Layers**: **Dropout**, **BatchNormalization**, Embedding, SelfAttention
- **Data Pipeline**: **DataLoader** with batching & shuffling, Dataset manifest
- **Loss functions**: MSE, MAE, CrossEntropy
- **CLI**: Command-line interface for ingestion, training, and export workflows
- **Cross-language format**: Save/load models in `.lle` v1.1 format

## Installation

```bash
npm install @students-dev/train-lle
# or
pnpm add @students-dev/train-lle
```

## Quickstart

### Programmatic Usage

```ts
import { Model, MLP, Trainer, Dataset, DataLoader, AdamW, Metrics } from "@students-dev/train-lle";

// Define a simple MLP
const model = new Model(MLP.build({ input: 4, layers: [16, 16], output: 1 }));

// Configure Trainer with AdamW and metrics
const trainer = new Trainer({ 
  optimizer: new AdamW(0.01), 
  loss: new MSE(), 
  epochs: 50 
});

// Load Data
const dataset = await Dataset.fromCSV("data.csv");
const loader = new DataLoader(dataset, { batchSize: 32, shuffle: true });

// Train
await trainer.fit(model, dataset.inputs, dataset.targets);

// Predict
const prediction = model.predict(new Tensor([1, 2, 3, 4], [1, 4]));
console.log(prediction);

// Save
await model.save("model.lle");
```

### Advanced Usage (Transformer)

```ts
import { TransformerClassifier, Trainer, AdamW } from "@students-dev/train-lle";

// Build a text classifier
const layers = TransformerClassifier.build({
  vocabSize: 10000,
  embedSize: 128,
  numBlocks: 2,
  classes: 5
});
const model = new Model(layers);

// Train...
```

### CLI Usage

```bash
# Ingest documents
npx train-lle ingest /path/to/documents

# Extract text
npx train-lle extract artifacts.json

# Assemble dataset
npx train-lle assemble-dataset manifest.json

# Train a model
npx train-lle train config.json

# Export model
npx train-lle export output.lle
```

## API Reference

### Models & Layers

- `MLP`, `CNN`, `RNN`
- `ResNet`, `TransformerClassifier`
- `Dense`, `Conv2D`, `Dropout`, `BatchNormalization`, `Embedding`, `SelfAttention`

### Core Classes

- `Model`: Neural network model
- `Trainer`: Training orchestrator with checkpoints
- `Dataset`, `DataLoader`: Data loading and preprocessing
- `Tensor`: Multi-dimensional array operations
- `Metrics`: Accuracy, MSE, MAE
- `AdamW`, `StepLR`, `CosineAnnealing`

## Examples

See `examples/` directory for runnable scripts:

- `examples/tabular/`: Regression on synthetic data
- `examples/image/`: Classification on tiny images
- `examples/text/`: Simple text classification

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

Apache-2.0