# train-lle

Python package for the Local Learning Engine.

## Installation

```bash
pip install train-lle
```

## Usage

```python
from train_lle import Model, MLP, Trainer, Dataset

model = Model(MLP.build(input=4, layers=[8, 8], output=1))
trainer = Trainer(optimizer="adam", lr=0.01, epochs=50, loss="mse")

dataset = Dataset.from_csv("data.csv")
trainer.fit(model, dataset.inputs, dataset.targets)

model.save("model.lle")
```