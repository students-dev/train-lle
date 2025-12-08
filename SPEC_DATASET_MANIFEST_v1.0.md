# Dataset Manifest Format v1.0

The DATASET_MANIFEST.json describes a dataset for training.

## Structure

- `version`: "1.0"
- `created`: ISO date
- `artifacts`: Array of {path, checksum, chunks}
- `chunks`: Array of {id, text, tokens, artifact}
- `splits`: {train, val, test} arrays of chunk ids

## Usage

Used by both JS and Python trainers for consistent dataset handling.