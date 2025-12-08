# CLI Usage

The train-lle CLI is available in all runtimes with identical commands.

## Commands

### init

Initialize a new project with a config file.

```bash
train-lle init
```

Creates `train-config.json` with default settings.

### train

Train a model using a config file or dataset.

```bash
train-lle train config.json
# or
train-lle train dataset.csv
```

### test

Test a model on a dataset.

```bash
train-lle test model.lle dataset.csv
```

### export

Export a model to a new file.

```bash
train-lle export model.lle output.lle
```

### stats

Show model statistics.

```bash
train-lle stats model.lle
```

## Config File

Example `train-config.json`:

```json
{
  "model": "mlp",
  "config": { "input": 4, "layers": [8, 8], "output": 1 },
  "optimizer": "adam",
  "lr": 0.01,
  "epochs": 50,
  "loss": "mse",
  "dataset": "data.csv"
}
```