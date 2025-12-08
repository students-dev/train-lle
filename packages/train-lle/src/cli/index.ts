import { Command } from "commander";
import { Model } from "../core/engine";
import { Trainer } from "../core/engine";
import { SGD, Adam } from "../core/optimizer";
import { MSE, CrossEntropy } from "../core/loss";
import { Dataset } from "../dataset/csv";
import { saveModel, loadModel } from "../export/lle-format";
import { MLP } from "../models/mlp";
import { discoverFiles } from "../ingest/index.js";
import { extractText } from "../extract/index.js";
import { normalizeText } from "../normalize/index.js";
import { tokenizeAndChunk } from "../tokenize/index.js";
import { assembleDataset } from "../dataset/manifest.js";
import { EmbeddingModel, VectorIndex } from "../embed/index.js";

const program = new Command();

program
  .name('train-lle')
  .description('Local Learning Engine CLI')
  .version('1.1.1');

program.command('init')
  .description('Initialize a new project')
  .action(() => {
    const config = {
      model: 'mlp',
      config: { input: 4, layers: [8, 8], output: 1 },
      optimizer: 'adam',
      lr: 0.01,
      epochs: 50,
      loss: 'mse',
      dataset: 'data.csv'
    };
    require('fs').writeFileSync('train-config.json', JSON.stringify(config, null, 2));
    console.log('Created train-config.json');
  });

program.command('train')
  .description('Train the model')
  .argument('<config>', 'config file or dataset path')
  .action(async (configPath) => {
    let config;
    if (configPath.endsWith('.json')) {
      config = JSON.parse(require('fs').readFileSync(configPath, 'utf-8'));
    } else {
      // assume dataset path, use default config
      config = { model: 'mlp', config: { input: 4, layers: [8, 8], output: 1 }, optimizer: 'adam', lr: 0.01, epochs: 50, loss: 'mse', dataset: configPath };
    }
    const dataset = Dataset.fromCSV(config.dataset);
    // Build model
    let model;
    if (config.model === 'mlp') {
      model = new Model(MLP.build(config.config));
    } else {
      throw new Error('Model not supported');
    }
    const optimizer = config.optimizer === 'adam' ? new Adam(config.lr) : new SGD(config.lr);
    const loss = config.loss === 'mse' ? new MSE() : new CrossEntropy();
    const trainer = new Trainer({ optimizer, loss, epochs: config.epochs });
    trainer.fit(model, dataset.inputs, dataset.targets);
    // Save model
    saveModel('model.lle', model);
    console.log('Training complete, model saved to model.lle');
  });

program.command('test')
  .description('Test the model')
  .argument('<model>', 'model file')
  .argument('<dataset>', 'dataset file')
  .action(async (modelPath, datasetPath) => {
    const model = await loadModel(modelPath);
    const dataset = Dataset.fromCSV(datasetPath);
    const trainer = new Trainer({ optimizer: new SGD(0), loss: new MSE(), epochs: 0 }); // dummy
    const loss = trainer.evaluate(model, dataset.inputs, dataset.targets);
    console.log(`Test loss: ${loss}`);
  });

program.command('export')
  .description('Export the model')
  .argument('<model>', 'model file')
  .argument('<output>', 'output file')
  .action(async (modelPath, outputPath) => {
    const model = await loadModel(modelPath);
    await saveModel(outputPath, model);
    console.log(`Model exported to ${outputPath}`);
  });

program.command('stats')
  .description('Show model statistics')
  .argument('<model>', 'model file')
  .action(async (modelPath) => {
    const model = await loadModel(modelPath);
    const params = model.params();
    let total = 0;
    for (const p of params) {
      total += p.data.length;
    }
    console.log(`Total parameters: ${total}`);
  });

program.command('ingest')
  .description('Ingest files from path')
  .argument('<path>', 'path to ingest')
  .action(async (path) => {
    const artifacts = await discoverFiles(path);
    console.log(`Ingested ${artifacts.length} files`);
    require('fs').writeFileSync('artifacts.json', JSON.stringify(artifacts, null, 2));
  });

program.command('extract')
  .description('Extract text from artifacts')
  .argument('<artifact>', 'artifact file')
  .action(async (artifactPath) => {
    const artifacts = JSON.parse(require('fs').readFileSync(artifactPath, 'utf-8'));
    const extracted = [];
    for (const art of artifacts) {
      const ext = await extractText(art);
      extracted.push(ext);
    }
    console.log(`Extracted ${extracted.length} artifacts`);
    require('fs').writeFileSync('extracted.json', JSON.stringify(extracted, null, 2));
  });

program.command('assemble-dataset')
  .description('Assemble dataset from extracted artifacts')
  .argument('<manifest>', 'manifest file')
  .action(async (manifestPath) => {
    const extracted = JSON.parse(require('fs').readFileSync('extracted.json', 'utf-8'));
    const normalized = extracted.map(normalizeText);
    const allChunks = [];
    for (const norm of normalized) {
      const chunks = tokenizeAndChunk(norm.text);
      allChunks.push(...chunks);
    }
    assembleDataset(normalized, allChunks, '.');
    console.log('Dataset assembled');
  });

program.command('index')
  .description('Index dataset')
  .argument('<dataset>', 'dataset path')
  .action(async (datasetPath) => {
    const manifest = JSON.parse(require('fs').readFileSync(`${datasetPath}/DATASET_MANIFEST.json`, 'utf-8'));
    const embedModel = new EmbeddingModel();
    const index = new VectorIndex();
    for (const chunk of manifest.chunks) {
      const vector = embedModel.embed(chunk.text);
      index.add(vector);
    }
    // Save index somehow, for now skip
    console.log('Indexing complete');
  });

program.command('train-from-corpus')
  .description('Train from corpus')
  .argument('<dataset>', 'dataset path')
  .action(async (datasetPath) => {
    // Similar to train, but from corpus
    console.log('Training from corpus not implemented yet');
  });

export function runCLI() {
  program.parse();
}