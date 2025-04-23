'use strict';

import * as utils from '../common/utils.js';
import { buildWebGL2Pipeline } from './lib/webgl2/webgl2Pipeline.js';
import * as ui from '../common/ui.js';
import { WebnnSelfieSegmenter } from './webnn_selfie_segmenter.js';

const imgElement = document.getElementById('feedElement');
imgElement.src = './images/test.jpg';
const camElement = document.getElementById('feedMediaElement');
const outputCanvas = document.getElementById('outputCanvas');
let rafReq;
let isFirstTimeLoad = true;
let inputType = 'image';
let stream = null;
let sess;
let wnnModel;
let loadTime = 0;
let computeTime = 0;
let outputBuffer;
let modelChanged = false;
let backgroundImageSource = document.getElementById('00-img');
let backgroundType = 'img'; // 'none', 'blur', 'image'
let modelType = 'webnn';
let deviceType = '';
let backend = 'webnn';

const inputOptions = {
  mean: [127.5, 127.5, 127.5],
  std: [127.5, 127.5, 127.5],
  scaledFlag: false,
  inputResolution: [256, 256],
};

const disabledSelectors = ['#tabs > li', '.btn'];


let gpuBuffer = null;
let gpuInputTensor = null;
function uploadToGPU(gpuBuffer, inputBuffer) {
  const stagingBuffer = ort.env.webgpu.device.createBuffer({usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
													 size: inputBuffer.byteLength,
													 mappedAtCreation: true});
  const arrayBuffer = stagingBuffer.getMappedRange();
  new Uint8Array(arrayBuffer).set(
    new Uint8Array(inputBuffer.buffer, inputBuffer.byteOffset, inputBuffer.byteLength),
  );
  stagingBuffer.unmap();
  const encoder = ort.env.webgpu.device.createCommandEncoder();
  encoder.copyBufferToBuffer(stagingBuffer, 0, gpuBuffer, 0, inputBuffer.byteLength);
  ort.env.webgpu.device.queue.submit([encoder.finish()]);
  stagingBuffer.destroy();
}
 
function createGpuTensorForInput(inputBuffer) {
  if (!gpuBuffer) {
    gpuBuffer = ort.env.webgpu.device.createBuffer({
      // eslint-disable-next-line no-bitwise
      usage:
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.STORAGE,
      size: Math.ceil(inputBuffer.byteLength / 16) * 16,
    });
  }
  uploadToGPU(gpuBuffer, inputBuffer);
 
  if (!gpuInputTensor) {
    gpuInputTensor = ort.Tensor.fromGpuBuffer(gpuBuffer, {
      dataType: 'float32',
      dims: inputOptions.inputShape,
      dispose: () => gpuBuffer.destroy(),
    });
  }
}

async function compute(modelType, inputBuffer) {
  console.time("compute function");
  let outputData;
  if (modelType == 'ort') {
    const feed = {};

    if (backend.includes('webgpu')) {
      createGpuTensorForInput(inputBuffer);
      feed['input'] = gpuInputTensor;
    } else {
      feed['input'] = new ort.Tensor(
        'float32',
        inputBuffer,
        inputOptions.inputShape,
      );
    }
    const result = await sess.run(feed);
    if (backend.includes('webgpu')) {
      outputData = await result['segment_back'].getData();
    } else {
      outputData = result['segment_back'].cpuData;
    }
  } else {
    outputData = await wnnModel.compute(inputBuffer);
  }
  console.timeEnd("compute function");
  return outputData

}

$(document).ready(async () => {
  $('.icdisplay').hide();
  ort.env.wasm.numThreads = 4;
});

$('#backend').on('change', async (e) => {
  modelChanged = true;
  backend = $(e.target).find(':selected').val();
  modelType = backend.includes('ort') ? 'ort' : 'webnn';
  if (!backend.includes('webnn') && deviceType != '') {
    $(`#${deviceType}`).parent().removeClass('active');
  }
  if (inputType === 'camera') utils.stopCameraStream(rafReq, stream);
  await main();
});

$('#deviceTypeBtns').on('change', async (e) => {
  modelChanged = true;
  deviceType = $(e.target).attr('id');
  if (inputType === 'camera') utils.stopCameraStream(rafReq, stream);
  await main();
});

// Click trigger to do inference with <img> element
$('#img').click(async () => {
  if (inputType === 'camera') utils.stopCameraStream(rafReq, stream);
  inputType = 'image';
  $('#pickimage').show();
  $('.shoulddisplay').hide();
  await main();
});

$('#imageFile').change((e) => {
  const files = e.target.files;
  if (files.length > 0) {
    $('#feedElement').removeAttr('height');
    $('#feedElement').removeAttr('width');
    imgElement.src = URL.createObjectURL(files[0]);
  }
});

$('#feedElement').on('load', async () => {
  await main();
});

// Click trigger to do inference with <video> media element
$('#cam').click(async () => {
  inputType = 'camera';
  $('#pickimage').hide();
  $('.shoulddisplay').hide();
  await main();
});

$('#gallery .gallery-item').click(async (e) => {
  $('#gallery .gallery-item').removeClass('hl');
  $(e.target).parent().addClass('hl');
  const backgroundTypeId = $(e.target).attr('id');
  backgroundImageSource = document.getElementById(backgroundTypeId);
  if (backgroundTypeId === 'no-img') {
    backgroundType = 'none';
  } else if (backgroundTypeId === 'blur-img') {
    backgroundType = 'blur';
  } else {
    backgroundType = 'image';
  }
  const srcElement = inputType == 'image' ? imgElement : camElement;
  await drawOutput(outputBuffer, srcElement);
});

/**
 * This method is used to render live camera tab.
 */
async function renderCamStream() {
  if (!stream.active) return;
  // If the video element's readyState is 0, the video's width and height are 0.
  // So check the readState here to make sure it is greater than 0.
  if (camElement.readyState === 0) {
    rafReq = requestAnimationFrame(renderCamStream);
    return;
  }
  const inputCanvas = utils.getVideoFrame(camElement);
  const inputBuffer = utils.getInputTensor(camElement, inputOptions);
  console.log('- Computing... ');

  const start = performance.now();
  outputBuffer = await compute(modelType, inputBuffer);
  computeTime = (performance.now() - start).toFixed(2);
  console.log(`  done in ${computeTime} ms.`);
  showPerfResult();
  await drawOutput(outputBuffer, inputCanvas);
  $('#fps').text(`${(1000 / computeTime).toFixed(0)} FPS`);
  rafReq = requestAnimationFrame(renderCamStream);
}

async function drawOutput(outputBuffer, srcElement) {
  outputCanvas.width = srcElement.width;
  outputCanvas.height = srcElement.height;
  const pipeline = buildWebGL2Pipeline(
    srcElement,
    backgroundImageSource,
    backgroundType,
    inputOptions.inputResolution,
    outputCanvas,
    outputBuffer
  );
  const postProcessingConfig = {
    smoothSegmentationMask: true,
    jointBilateralFilter: { sigmaSpace: 1, sigmaColor: 0.1 },
    coverage: [0.5, 0.75],
    lightWrapping: 0.3,
    blendMode: 'screen',
  };
  pipeline.updatePostProcessingConfig(postProcessingConfig);
  await pipeline.render();
}

function showPerfResult(medianComputeTime = undefined) {
  $('#loadTime').html(`${loadTime} ms`);
  if (medianComputeTime !== undefined) {
    $('#computeLabel').html('Median inference time:');
    $('#computeTime').html(`${medianComputeTime} ms`);
  } else {
    $('#computeLabel').html('Inference time:');
    $('#computeTime').html(`${computeTime} ms`);
  }
}

export async function main() {
  try {
    if (deviceType === '' && backend.includes('webnn')) return;
    ui.handleClick(disabledSelectors, true);
    if (isFirstTimeLoad) $('#hint').hide();
    const numRuns = utils.getUrlParams()[0];
    // Only do load() when model first time loads and
    // there's new model or delegate choosed
    if (isFirstTimeLoad || modelChanged) {
      modelChanged = false;
      isFirstTimeLoad = false;
      // UI shows model loading progress
      await ui.showProgressComponent('current', 'pending', 'pending');
      const start = performance.now();
      if (modelType == 'ort') {  
        const provider = backend.split('-')[1];
        console.log(`- Loading ORT model, provider: [${provider}], deviceType: [${deviceType}]`);
        const options = {
          executionProviders: [
            {
              name: provider,
              deviceType: deviceType,
              preferredLayout: provider == 'webgpu' ? 'NHWC' : undefined,
            },
          ],
          enableGraphCapture: provider == 'webgpu',
          graphOptimizationLevel: 'all',
          logSeverityLevel: 0,
        };
        inputOptions.inputLayout = 'nhwc';
        inputOptions.inputShape = [1, 256, 256, 3];
        sess = await ort.InferenceSession.create('./selfie_segmenter.onnx', options);
      } else {
        console.log(`- Loading WebNN model, deviceType: [${deviceType}]`);
        wnnModel = new WebnnSelfieSegmenter(deviceType);
        inputOptions.inputLayout = wnnModel.layout;
        inputOptions.inputShape = wnnModel.inputShape;
        const graph = await wnnModel.load({deviceType});
        await wnnModel.build(graph);
      }
      loadTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${loadTime} ms.`);
      // UI shows model building progress
      await ui.showProgressComponent('done', 'current', 'pending');
    }
    // UI shows inferencing progress
    await ui.showProgressComponent('done', 'done', 'current');
    if (inputType === 'image') {
      const inputBuffer = utils.getInputTensor(imgElement, inputOptions);
      console.log('- Computing... ');
      const computeTimeArray = [];
      let medianComputeTime;

      console.log('- Warmup... ');
     // outputBuffer = await compute(modelType, inputBuffer);
      console.log('- Warmup done... ');

      for (let i = 0; i < numRuns; i++) {
        const start = performance.now();
        outputBuffer = await compute(modelType, inputBuffer);
        const time = performance.now() - start;
        console.log(`  compute time ${i + 1}: ${time.toFixed(2)} ms`);
        computeTimeArray.push(time);
      }
      computeTime = utils.getMedianValue(computeTimeArray);
      computeTime = computeTime.toFixed(2);
      if (numRuns > 1) {
        medianComputeTime = computeTime;
      }

      console.log('outputBuffer: ', outputBuffer);

      await ui.showProgressComponent('done', 'done', 'done');
      $('#fps').hide();
      ui.readyShowResultComponents();
      await drawOutput(outputBuffer, imgElement);
      showPerfResult(medianComputeTime);
    } else if (inputType === 'camera') {
      stream = await utils.getMediaStream();
      camElement.srcObject = stream;
      camElement.onloadedmediadata = await renderCamStream();
      await ui.showProgressComponent('done', 'done', 'done');
      $('#fps').show();
      ui.readyShowResultComponents();
    } else {
      throw Error(`Unknown inputType ${inputType}`);
    }
  } catch (error) {
    console.log(error);
    ui.addAlert(error.message);
  }
  ui.handleClick(disabledSelectors, false);
}
