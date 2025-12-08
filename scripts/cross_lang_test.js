// Cross-language test: Load Python-trained model in JS
import { loadModel } from 'train-lle';

async function test() {
  const model = await loadModel('model_py.lle');
  const input = [1, 2, 3, 4]; // example
  const pred = model.predict(input);
  console.log('Prediction:', pred.data);
  // Verify against expected
  const expected = [0.5, 0.6]; // placeholder
  if (Math.abs(pred.data[0] - expected[0]) < 0.01) {
    console.log('Test passed');
  } else {
    console.log('Test failed');
  }
}

test();