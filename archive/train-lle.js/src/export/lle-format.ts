import { createWriteStream, createReadStream } from "fs";
import { Model, Layer } from "../core/engine";
import { Tensor } from "../core/tensor";
import { Dense, ReLU, Sigmoid, Linear } from "../models/mlp";
import { Conv2D, Flatten } from "../models/cnn";
import { RNN } from "../models/rnn";
import JSZip from "jszip";

function createLayer(type: string, config: any): Layer {
  switch (type) {
    case "dense":
      return new Dense(config.inputSize, config.outputSize);
    case "relu":
      return new ReLU();
    case "sigmoid":
      return new Sigmoid();
    case "linear":
      return new Linear();
    case "conv2d":
      return new Conv2D(config.inChannels, config.outChannels, config.kernelSize);
    case "flatten":
      return new Flatten();
    case "rnn":
      return new RNN(config.inputSize, config.hiddenSize);
    default:
      throw new Error(`Unknown layer type: ${type}`);
  }
}

export async function saveModel(path: string, model: Model): Promise<void> {
  const zip = new JSZip();

  // metadata.json
  zip.file("metadata.json", JSON.stringify({
    version: "1.1",
    format: "lle"
  }));

  // graph.json
  const graph = model.layers.map(layer => ({
    type: layer.type,
    config: layer.config
  }));
  zip.file("graph.json", JSON.stringify(graph));

  // weights.bin
  const params = model.params();
  const totalSize = params.reduce((sum, p) => sum + p.data.length, 0) * 4;
  const buffer = Buffer.alloc(totalSize);
  let offset = 0;
  const weightsIndex: { shape: number[]; offset: number; size: number }[] = [];
  for (const param of params) {
    const size = param.data.length * 4;
    for (let i = 0; i < param.data.length; i++) {
      buffer.writeFloatLE(param.data[i], offset + i * 4);
    }
    weightsIndex.push({ shape: param.shape, offset, size });
    offset += size;
  }
  zip.file("weights.bin", buffer);

  // weights_index.json
  zip.file("weights_index.json", JSON.stringify(weightsIndex));

  // config.json (optional)
  zip.file("config.json", JSON.stringify({}));

  const content = await zip.generateAsync({ type: "nodebuffer" });
  require("fs").writeFileSync(path, content);
}

export async function loadModel(path: string): Promise<Model> {
  const zip = new JSZip();
  const data = require("fs").readFileSync(path);
  await zip.loadAsync(data);

  // graph.json
  const graphData = await zip.file("graph.json")!.async("text");
  const graph = JSON.parse(graphData);

  // weights_index.json
  const indexData = await zip.file("weights_index.json")!.async("text");
  const weightsIndex = JSON.parse(indexData);

  // weights.bin
  const weightsBuffer = await zip.file("weights.bin")!.async("nodebuffer");

  const layers: Layer[] = [];
  let paramIdx = 0;
  for (const layerDef of graph) {
    const layer = createLayer(layerDef.type, layerDef.config);
    const layerParams = layer.params();
    for (const param of layerParams) {
      const idx = weightsIndex[paramIdx];
      const arr = new Float32Array(idx.size / 4);
      for (let i = 0; i < arr.length; i++) {
        arr[i] = weightsBuffer.readFloatLE(idx.offset + i * 4);
      }
      // Set the param
      if (paramIdx < weightsIndex.length) {
        // Assuming order matches
        // This is hacky, but for simplicity
        Object.assign(param, { data: arr, shape: idx.shape });
      }
      paramIdx++;
    }
    layers.push(layer);
  }

  return new Model(layers);
}