'use strict';

importScripts('./mobilenet_nhwc_sync.js');
importScripts('../common/utils_worker.js');

let netInstance = null;
let outputOperand = null;
let isWebNN = true;
if (typeof MLGraphBuilder == 'undefined') {
  isWebNN = false;
  // WebNN is not supported, use webnn-polyfill instead
  importScripts('https://cdn.jsdelivr.net/npm/@webmachinelearning/webnn-polyfill/dist/webnn-polyfill.js');
}

// Receive the message from the main thread
onmessage = async (message) => {
  if (message) {
    //Load WebNN graph
    switch (message.data.action) {
      case 'load':
        const loatStart = performance.now();
        const contextOptions = message.data.options;
        if (!isWebNN) {
          // Set WebNN polyfill backend
          await setPolyfillBackend(contextOptions.deviceType);
        }
        netInstance = new MobileNetV2NhwcSync();
        outputOperand = await netInstance.load(contextOptions);
        const loadTime = (performance.now() - loatStart).toFixed(2);
        postMessage(loadTime);
        break;

      case 'build':
        const buildStart = performance.now();
        netInstance.build(outputOperand);
        const buildTime = (performance.now() - buildStart).toFixed(2);
        postMessage(buildTime);
        break;

      case 'compute':
        const inputBuffer = message.data.buffer;
        const inputType = message.data.options.inputType;
        const numRuns = message.data.options.numRuns || 1;
        const outputBuffer = new Float32Array(sizeOfShape(netInstance.outputDimensions));
        let computeTime, computeTimeArray = [];
        if(inputType == 'image') {
          // Do warm up
          netInstance.compute(inputBuffer, outputBuffer);
        }
        for (let i = 0; i < numRuns; i++) {
          const computeStart = performance.now();
          netInstance.compute(inputBuffer, outputBuffer);
          computeTime = (performance.now() - computeStart).toFixed(2);
          console.log(`  compute time ${i + 1}: ${computeTime} ms`);
          computeTimeArray.push(Number(computeTime));
        }
        if (numRuns > 1) {
          computeTime = getMedianValue(computeTimeArray);
          computeTime = computeTime.toFixed(2);
          console.log(`  median compute time: ${computeTime} ms`);
        }
        postMessage({outputBuffer, computeTime}, [outputBuffer.buffer]);
        break;

      default:
        break;
    }
  }
};
