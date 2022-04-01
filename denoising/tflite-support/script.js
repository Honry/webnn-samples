MODEL_PATH = 'model.tflite';

async function load() {
  //////////////////////////////////////////////////////////////////////////////
  // Create the model runner with the model.

  const startTs = Date.now();

  // Load WASM module and model.
  const [module, modelArrayBuffer] = await Promise.all([
    tflite_model_runner_ModuleFactory(),
    (await fetch(MODEL_PATH)).arrayBuffer(),
  ]);
  const modelBytes = new Uint8Array(modelArrayBuffer);
  const offset = module._malloc(modelBytes.length);
  module.HEAPU8.set(modelBytes, offset);

  // Create model runner.
  const modelRunnerResult =
    module.TFLiteWebModelRunner.CreateFromBufferAndOptions(
      offset, modelBytes.length, {
      numThreads: Math.min(
        4, Math.max(1, (navigator.hardwareConcurrency || 1) / 2)),
      enableWebNNDelegate: false,
      webNNDevicePreference: 0 // 0 - default, 1 - gpu, 2 - cpu
    });
  if (!modelRunnerResult.ok()) {
    throw new Error(
      'Failed to create TFLiteWebModelRunner: ' + modelRunner.errorMessage());
  }
  const modelRunner = modelRunnerResult.value();
  const loadFinishedMs = Date.now() - startTs;
  document.querySelector('.loading-stats').textContent =
    `Loaded WASM module and TFLite model ${MODEL_PATH} in ${loadFinishedMs}ms`;
  document.querySelector('.hide').classList.remove('hide');
  return modelRunner;

}

let modelRunner;
window.onload = async function () {
  modelRunner = await load();
}

function start() {

  //////////////////////////////////////////////////////////////////////////////
  // Get input and output info.

  const inputs = callAndDelete(
    modelRunner.GetInputs(), results => convertCppVectorToArray(results));
  const outputs = callAndDelete(
    modelRunner.GetOutputs(), results => convertCppVectorToArray(results));

  //////////////////////////////////////////////////////////////////////////////
  // Set input tensor data.
  // Inputs
  // name: generate_mask_initial_states_14:0 type: float32[1,128]
  // name: generate_mask_initial_states_11:0 type: float32[1,128]
  // name: generate_mask_initial_states_9:0 type: float32[1,128]
  // name: generate_mask_initial_states_1:0 type: float32[1,128]
  // name: generate_mask_initial_states_5:0 type: float32[1,128]
  // name: generate_mask_initial_states_3:0 type: float32[1,128]
  // name: generate_mask_initial_states_18:0 type: float32[1,128]
  // name: generate_mask_initial_states_20:0 type: float32[1,128]
  // name: generate_mask_initial_states_19:0 type: float32[1,128]
  // name: generate_mask_initial_states_27:0 type: float32[1,128]
  // name: generate_mask_initial_states_17:0 type: float32[1,128]
  // name: generate_mask_initial_states_0:0 type: float32[1,128]
  // name: generate_mask_initial_states_28:0 type: float32[1,128]
  // name: generate_mask_initial_states_26:0 type: float32[1,128]
  // name: generate_mask_initial_states_22:0 type: float32[1,128]
  // name: generate_mask_initial_states_29:0 type: float32[1,128]
  // name: generate_mask_initial_states_30:0 type: float32[1,128]
  // name: generate_mask_initial_states_10:0 type: float32[1,128]
  // name: generate_mask_compressed_spectrograms:0 type: float32[1,1,1,257]
  // name: generate_mask_initial_states_7:0 type: float32[1,128]
  // name: generate_mask_initial_states_15:0 type: float32[1,128]
  // name: generate_mask_initial_states_31:0 type: float32[1,128]
  // name: generate_mask_initial_states_8:0 type: float32[1,128]
  // name: generate_mask_initial_states_12:0 type: float32[1,128]
  // name: generate_mask_initial_states_2:0 type: float32[1,128]
  // name: generate_mask_initial_states_24:0 type: float32[1,128]
  // name: generate_mask_initial_states_21:0 type: float32[1,128]
  // name: generate_mask_initial_states_13:0 type: float32[1,128]
  // name: generate_mask_initial_states_16:0 type: float32[1,128]
  // name: generate_mask_initial_states_23:0 type: float32[1,128]
  // name: generate_mask_initial_states_4:0 type: float32[1,128]
  // name: generate_mask_initial_states_6:0 type: float32[1,128]
  // name: generate_mask_initial_states_25:0 type: float32[1,128]
  console.log("Inputs length: ", inputs.length);
  for (let i = 0; i < inputs.length; i ++) {
    const inputBuffer = inputs[i].data();
    const inputData = new Float32Array(inputBuffer.length);
    inputData.fill(1.5);
    console.log(inputData);
    inputBuffer.set(inputData);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Infer, get output tensor, and sort by logit values in reverse.

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
    const success = modelRunner.Infer();
    const inferTime = (performance.now() - start).toFixed(2);
    if (!success) return;
    console.log(`Infer time ${i + 1}: ${inferTime} ms`);
    inferTimes.push(Number(inferTime));
  }
  console.log("outputs: ", outputs);
  const result = Array.from(outputs[0].data());
  console.log("outputs 0 : ", result);
  //////////////////////////////////////////////////////////////////////////////
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

function getMedianValue(array) {
  array = array.sort((a, b) => a - b);
  const medianValue = array.length % 2 !== 0 ? array[Math.floor(array.length / 2)] :
    (array[array.length / 2 - 1] + array[array.length / 2]) / 2;
  return medianValue.toFixed(2);
}


////////////////////////////////////////////////////////////////////////////////
// Helper functions.

/** Converts the given c++ vector to a JS array. */
function convertCppVectorToArray(vector) {
  if (vector == null) return [];

  const result = [];
  for (let i = 0; i < vector.size(); i++) {
    const item = vector.get(i);
    result.push(item);
  }
  return result;
}

/**
 * Calls the given function with the given deletable argument, ensuring that
 * the argument gets deleted afterwards (even if the function throws an error).
 */
function callAndDelete(arg, func) {
  try {
    return func(arg);
  } finally {
    if (arg != null) arg.delete();
  }
}
// outputs
// name: StatefulPartitionedCall_1:32 type: float32[1,128]
// name: StatefulPartitionedCall_1:27 type: float32[1,128]
// name: StatefulPartitionedCall_1:1 type: float32[1,128]
// name: StatefulPartitionedCall_1:22 type: float32[1,128]
// name: StatefulPartitionedCall_1:21 type: float32[1,128]
// name: StatefulPartitionedCall_1:16 type: float32[1,128]
// name: StatefulPartitionedCall_1:10 type: float32[1,128]
// name: StatefulPartitionedCall_1:13 type: float32[1,128]
// name: StatefulPartitionedCall_1:30 type: float32[1,128]
// name: StatefulPartitionedCall_1:6 type: float32[1,128]
// name: StatefulPartitionedCall_1:4 type: float32[1,128]
// name: StatefulPartitionedCall_1:28 type: float32[1,128]
// name: StatefulPartitionedCall_1:23 type: float32[1,128]
// name: StatefulPartitionedCall_1:15 type: float32[1,128]
// name: StatefulPartitionedCall_1:14 type: float32[1,128]
// name: StatefulPartitionedCall_1:20 type: float32[1,128]
// name: StatefulPartitionedCall_1:25 type: float32[1,128]
// name: StatefulPartitionedCall_1:5 type: float32[1,128]
// name: StatefulPartitionedCall_1:18 type: float32[1,128]
// name: StatefulPartitionedCall_1:19 type: float32[1,128]
// name: StatefulPartitionedCall_1:9 type: float32[1,128]
// name: StatefulPartitionedCall_1:11 type: float32[1,128]
// name: StatefulPartitionedCall_1:0 type: float32[1,2,1,257]
// name: StatefulPartitionedCall_1:17 type: float32[1,128]
// name: StatefulPartitionedCall_1:12 type: float32[1,128]
// name: StatefulPartitionedCall_1:7 type: float32[1,128]
// name: StatefulPartitionedCall_1:31 type: float32[1,128]
// name: StatefulPartitionedCall_1:3 type: float32[1,128]
// name: StatefulPartitionedCall_1:2 type: float32[1,128]
// name: StatefulPartitionedCall_1:8 type: float32[1,128]
// name: StatefulPartitionedCall_1:26 type: float32[1,128]
// name: StatefulPartitionedCall_1:29 type: float32[1,128]
// name: StatefulPartitionedCall_1:24 type: float32[1,128]