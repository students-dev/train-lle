# LLE Format Specification v1.1

The .lle format is a cross-language model serialization format for train-lle.

## Structure

The .lle file is a ZIP archive containing:

- `metadata.json`: Version and format info
- `graph.json`: Model architecture
- `weights_index.json`: Weight offsets and shapes
- `weights.bin`: Binary weights data
- `config.json`: Optional configuration

## Data Types

- All floats are Float32 little-endian
- Weights are stored as contiguous Float32 arrays

## Compatibility

- Train in JS, run in Python
- Train in Python, run in JS
- Deterministic file ordering