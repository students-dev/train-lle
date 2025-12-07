# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.html).

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