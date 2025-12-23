// Top-level exports for train-lle

export { Tensor } from "./core/tensor";
export { Model, Trainer, Checkpoint } from "./core/engine";
export { Dataset } from "./dataset/csv";
export { MLP, ReLU, Sigmoid, Tanh, Softmax, Linear, Dropout, BatchNormalization } from "./models/mlp";
export { CNN } from "./models/cnn";
export { RNN } from "./models/rnn";
export { SGD, Adam, RMSProp, AdamW } from "./core/optimizer";
export { StepLR, CosineAnnealing } from "./core/scheduler";
export { MSE, MAE, CrossEntropy } from "./core/loss";
export { Metrics } from "./core/metrics";
export { DataLoader } from "./dataset/dataloader";
export { ResNet, ResidualBlock } from "./models/resnet";
export { TransformerClassifier, TransformerBlock, SelfAttention, Embedding, GlobalAveragePooling1D } from "./models/transformer";
export { saveModel, loadModel } from "./export/lle-format";
export { runCLI as cli } from "./cli/index";

// Namespace export
import { Model, Trainer, Checkpoint } from "./core/engine";
import { Dataset } from "./dataset/csv";
import { DataLoader } from "./dataset/dataloader";
import { runCLI as cli } from "./cli/index";
import { Dropout, BatchNormalization } from "./models/mlp";
import { StepLR, CosineAnnealing } from "./core/scheduler";
import { ResNet } from "./models/resnet";
import { TransformerClassifier } from "./models/transformer";
import { Metrics } from "./core/metrics";

export const LLE = {
  Model,
  Trainer,
  Checkpoint,
  Dataset,
  DataLoader,
  Metrics,
  cli,
  Dropout,
  BatchNormalization,
  StepLR,
  CosineAnnealing,
  ResNet,
  TransformerClassifier
};