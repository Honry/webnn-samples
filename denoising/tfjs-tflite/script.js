MODEL_PATH = 'model.tflite';

async function load() {
  const numThreads = parseInt(document.getElementById('numThreads').value);
  document.getElementById('predict').hidden = true;
  document.querySelector('.result').innerHTML = "";
  const startTs = Date.now();
  const tfliteModel = await tflite.loadTFLiteModel(MODEL_PATH, {
    numThreads: numThreads,
  });
  const loadFinishedMs = Date.now() - startTs;
  document.querySelector('.loading-stats').textContent =
    `Loaded WASM module and TFLite model ${MODEL_PATH} with ${numThreads} threads in ${loadFinishedMs}ms`;
  document.getElementById('predict').hidden = false;
  return tfliteModel;
}

let tfliteModel;
const loadElem = document.getElementById('loadModel');
loadElem.onclick = async function () {
  document.querySelector('.loading-stats').textContent = "Loading...";
  tfliteModel = await load();
}

async function start() {
  let inputs = new Array();
  const modelInputs = tfliteModel.inputs;
  // Set input tensor data.
  for (let i = 0; i < modelInputs.length; i ++) {
    const inputTensor = tf.randomNormal(modelInputs[i].shape);
    inputs.push(inputTensor);
  }

  // Set 'numRuns' param to run inference multiple times
  // numRuns includes the first run of inference

  let numRuns = document.getElementById('numRuns').value;
  console.log('numRuns: ', numRuns);

  if (numRuns < 1) {
    alert('Run Number should be greater than 0!');
    return;
  }
  numRuns = numRuns === null ? 1 : parseInt(numRuns);

  const inferTimes = [];
  for (let i = 0; i < numRuns; i++) {
    const start = performance.now();
    const outputs = tfliteModel.predict(inputs);
    const inferTime = (performance.now() - start).toFixed(2);
    console.log(`Infer time ${i + 1}: ${inferTime} ms`);
    inferTimes.push(Number(inferTime));
  }

  // Show result.
  if (inferTimes.length > 1) {
    const averageTime = (inferTimes.reduce((acc, curr) => acc + curr, 0) / inferTimes.length).toFixed(2);
    const minTime = Math.min(...inferTimes);
    const maxTime = Math.max(...inferTimes);
    const medianTime = getMedianValue(inferTimes);
    document.querySelector('.result').innerHTML =
      ` numRuns: ${numRuns} <br> average time: ${averageTime} ms <br>
      median time: ${medianTime} ms <br> max Time: ${maxTime} ms <br>
      min Time: ${minTime} ms`;
  } else {
    document.querySelector('.result').textContent =
      ` latency: ${inferTimes[0]}ms)`;
  }
}

// Helper functions.
function getMedianValue(array) {
  array = array.sort((a, b) => a - b);
  const medianValue = array.length % 2 !== 0 ? array[Math.floor(array.length / 2)] :
    (array[array.length / 2 - 1] + array[array.length / 2]) / 2;
  return medianValue.toFixed(2);
}


