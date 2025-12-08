// Cross-language test: Load Python-trained model in JS
const { loadModel } = require('train-lle.js');

async function test() {
  const model = await loadModel('model_py.lle');
  const input = [1, 2, 3, 4]; // example
  const pred = model.forward(input);
  console.log('Prediction:', pred);
  // Verify against expected
  const expected = [0.5, 0.6]; // placeholder
  if (Math.abs(pred[0] - expected[0]) < 0.01) {
    console.log('Test passed');
  } else {
    console.log('Test failed');
  }
}

test();