import { describe, it, expect } from "vitest";
import { Model } from "../../src/core/engine";
import { Trainer } from "../../src/core/engine";
import { MLP } from "../../src/models/mlp";
import { SGD } from "../../src/core/optimizer";
import { MSE } from "../../src/core/loss";
import { Tensor } from "../../src/core/tensor";

describe('Integration', () => {
  it('should train and reduce loss', () => {
    const inputs = new Tensor([1,0,0,0, 0,1,0,0], [2,4]);
    const targets = new Tensor([1, 0], [2,1]);
    const model = new Model(MLP.build({ input: 4, layers: [4], output: 1 }));
    const trainer = new Trainer({ optimizer: new SGD(0.1), loss: new MSE(), epochs: 10 });
    const initialLoss = trainer.evaluate(model, inputs, targets);
    trainer.fit(model, inputs, targets);
    const finalLoss = trainer.evaluate(model, inputs, targets);
    expect(finalLoss).toBeLessThan(initialLoss);
  });
});