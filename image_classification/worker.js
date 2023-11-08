'use strict';

importScripts('./mobilenet_nhwc_sync.js');
importScripts('./mobilenet_nchw_sync.js');
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
        const contextOptions = message.data.options.contextOptions;
        const layout = message.data.options.layout;
        if (!isWebNN) {
          // Set WebNN polyfill backend
          await setPolyfillBackend(contextOptions.deviceType);
        }
        netInstance = layout == 'nhwc' ? new MobileNetV2NhwcSync() : new MobileNetV2NchwSync();
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
        const outputBuffer = new Float32Array(sizeOfShape(netInstance.outputDimensions));
        const computeStart = performance.now();
        netInstance.compute(inputBuffer, outputBuffer);
        const computeTime = (performance.now() - computeStart).toFixed(2);
        console.log(`  compute time inside worker: ${computeTime} ms`);
        postMessage({outputBuffer}, [outputBuffer.buffer]);
        break;

      default:
        break;
    }
  }
};
