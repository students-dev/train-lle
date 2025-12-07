import { Command } from "commander";
import { Model } from "../core/engine";
import { Trainer } from "../core/engine";
import { SGD, Adam } from "../core/optimizer";
import { MSE, CrossEntropy } from "../core/loss";
import { Dataset } from "../dataset/csv";
import { saveModel, loadModel } from "../export/lle-format";
import { MLP } from "../models/mlp";

const program = new Command();

program
  .name('train-lle')
  .description('Local Learning Engine CLI')
  .version('0.1.0');

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
  .action((modelPath, datasetPath) => {
    const model = loadModel(modelPath);
    const dataset = Dataset.fromCSV(datasetPath);
    const trainer = new Trainer({ optimizer: new SGD(0), loss: new MSE(), epochs: 0 }); // dummy
    const loss = trainer.evaluate(model, dataset.inputs, dataset.targets);
    console.log(`Test loss: ${loss}`);
  });

program.command('export')
  .description('Export the current model')
  .argument('<output>', 'output file')
  .action(() => {
    // Assume model is in memory or load from config
    console.log('Export not implemented yet');
  });

program.command('stats')
  .description('Show model statistics')
  .argument('<model>', 'model file')
  .action((modelPath) => {
    const model = loadModel(modelPath);
    const params = model.params();
    let total = 0;
    for (const p of params) {
      total += p.data.length;
    }
    console.log(`Total parameters: ${total}`);
  });

export function runCLI() {
  program.parse();
}