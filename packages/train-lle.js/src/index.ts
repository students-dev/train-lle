// Top-level exports for train-lle

export { Tensor } from "./core/tensor";
export { Model, Trainer } from "./core/engine";
export { Dataset } from "./dataset/csv";
export { MLP, ReLU, Sigmoid, Tanh, Softmax, Linear } from "./models/mlp";
export { CNN } from "./models/cnn";
export { RNN } from "./models/rnn";
export { SGD, Adam, RMSProp } from "./core/optimizer";
export { MSE, MAE, CrossEntropy } from "./core/loss";
export { saveModel, loadModel } from "./export/lle-format";
export { runCLI as cli } from "./cli/index";

// Namespace export
import { Model, Trainer } from "./core/engine";
import { Dataset } from "./dataset/csv";
import { runCLI as cli } from "./cli/index";

export const LLE = {
  Model,
  Trainer,
  Dataset,
  cli,
};