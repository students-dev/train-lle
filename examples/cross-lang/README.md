# Cross-Language Usage Examples

Train a model in one runtime and load in another.

## Train in JavaScript, Load in Python

### Train in JS

```js
const { Model, MLP, Trainer, Dataset } = require("train-lle.js");

const model = new Model(MLP.build({ input: 4, layers: [8, 8], output: 1 }));
const trainer = new Trainer({ optimizer: "adam", lr: 0.01, epochs: 50, loss: "mse" });

const dataset = Dataset.fromCSV("data.csv");
trainer.fit(model, dataset.inputs, dataset.targets);

model.save("model_js.lle");
```

### Load in Python

```python
import train_lle

model = train_lle.load_model("model_js.lle")
prediction = model.predict([1, 2, 3, 4])
print(prediction)
```

## Train in Python, Load in JavaScript

### Train in Python

```python
from train_lle import Model, MLP, Trainer, Dataset

model = Model(MLP.build(input=4, layers=[8, 8], output=1))
trainer = Trainer(optimizer="adam", lr=0.01, epochs=50, loss="mse")

dataset = Dataset.from_csv("data.csv")
trainer.fit(model, dataset.inputs, dataset.targets)

model.save("model_py.lle")
```

### Load in JS

```js
import { loadModel } from "train-lle";

const model = await loadModel("model_py.lle");
const prediction = model.predict([1, 2, 3, 4]);
console.log(prediction);
```