# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0/).

## [1.2.0] - 2025-12-23

### Added
- **New Advanced Models**: Added implementations for **ResNet** (Vision) and **Transformer** (NLP).
- **New Layers**: `Dropout`, `BatchNormalization`, `GlobalAveragePooling1D`, `Embedding`, `SelfAttention`.
- **Enhanced Data Loading**: `DataLoader` class with batching, shuffling, and drop-last support.
- **New Optimizers**: `AdamW` with decoupled weight decay.
- **Learning Rate Schedulers**: `StepLR`, `CosineAnnealing`.
- **Training Checkpoints**: Automatically save best models based on validation loss.
- **Metrics**: Built-in support for Accuracy, MSE, and MAE.
- **Project Structure**: Added `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` and GitHub Issue Templates.

### Changed
- **Unified Package**: Consolidated TS/JS packages into a single `@students-dev/train-lle` package supporting both ESM and CommonJS.
- **Repository Organization**: Reorganized source code into `utils/math`, `utils/data`, etc.
- **Removed**: Archived Python support to focus on Node.js ecosystem (moved to `archive/`).

## [1.0.0] - 2025-12-08

### Added
- **Major release**: Full production-ready LOCAL LEARNING ENGINE ecosystem
- **Cross-language support**: TypeScript (ESM), JavaScript (CommonJS), Python (PyPI) packages
- **Cross-language compatibility**: Identical model behavior, training results, and .lle v1.1 format across all runtimes
- **Enhanced .lle format v1.1**: ZIP container with metadata.json, graph.json, weights_index.json, weights.bin
- **Additional optimizers**: RMSProp
- **Additional loss functions**: MAE
- **Additional activations**: Tanh, Softmax
- **Mono-repo structure**: Packages in packages/ directory with shared CI/CD
- **Cross-language testing**: Scripts to verify model compatibility between JS and Python
- **Updated CLI**: Consistent commands across runtimes
- **Comprehensive documentation**: SPEC_LLE_v1.1.md, updated READMEs, examples

### Changed
- Restructured repo into mono-repo with packages/train-lle/, train-lle.js/, train-lle-py/
- Updated build systems for all packages
- Enhanced CI to test all runtimes and cross-language compatibility

## [0.1.0] - 2025-12-07

### Added
- Initial MVP release of train-lle, a local-first ML training & inference engine for Node.js
- Core tensor operations with Float32Array backend
- Basic math operations: matrix multiply, add, transpose
- Optimizers: SGD and Adam
- Loss functions: MSE and CrossEntropy
- Training engine with forward, backward, and parameter updates
- Models: MLP (Dense), CNN, RNN
- Dataset parsers: CSV, JSON, basic image helpers
- Custom .lle format for model save/load
- CLI tool with init, train, test, export, stats commands
- Examples for tabular, image, and text data
- Unit and integration tests
- TypeScript types and documentation