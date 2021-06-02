'use strict';

import {FastStyleTransferNet} from './fast_style_transfer_net.js';
import {showProgressComponent, readyShowResultComponents} from './ui.js';

const maxWidth = 380;
const maxHeight = 380;
const imgElement = document.getElementById('feedElement');
imgElement.src = './images/content-images/travelspace.jpg';
const camElement = document.getElementById('feedMediaElement');
let modelId = 'starry-night';
let isFirstTimeLoad = true;
let isModelChanged = false;
let shouldStopFrame = false;
let inputType = 'image';
let fastStyleTransferNet;
let stream = null;
let loadTime = 0;
let buildTime = 0;
let computeTime = 0;
let firstInferenceTime = 0;
let useWebnn = false;
const inputOptions = {
  inputDimensions: [1, 540, 540, 3],
  inputLayout: 'nhwc',
}

$(document).ready(() => {
  $('.icdisplay').hide();
  $('.badge').html(modelId);
});

// Click trigger to do inference with <img> element
$('#img').click(async () => {
  shouldStopFrame = true;
  if (stream !== null) {
    stopCamera();
  }
  inputType = 'image';
  $('.shoulddisplay').hide();
  await main();
});

$('#gallery .gallery-image').hover((e) => {
  const id = $(e.target).attr('id');
  const modelName = $('#' + id).attr('title');
  $('.badge').html(modelName);
}, () => {
  const modelName = $(`#${modelId}`).attr('title');
  $('.badge').html(modelName);
});

$('#imageFile').change((e) => {
  const files = e.target.files;
  if (files.length > 0) {
    $('#feedElement').on('load', async () => {
      await main();
    });
    $('#feedElement').removeAttr('height');
    $('#feedElement').removeAttr('width');
    imgElement.src = URL.createObjectURL(files[0]);
  }
});

// Click trigger to do inference with <video> media element
$('#cam').click(async () => {
  inputType = 'camera';
  $('.shoulddisplay').hide();
  await main();
});

// Click handler to do inference with switched <img> element
async function handleImageSwitch(e) {
  const newModelId = $(e.target).attr('id');
  if (newModelId !== modelId) {
    shouldStopFrame = true;
    isModelChanged = true;
    modelId = newModelId;
    const modelName = $(`#${modelId}`).attr('title');
    $('.badge').html(modelName);
    $('#gallery .gallery-item').removeClass('hl');
    $(e.target).parent().addClass('hl');
  }
  await main();
}

async function getMediaStream() {
  // Support 'user' facing mode at present
  const constraints = {audio: false, video: {facingMode: 'user'}};
  stream = await navigator.mediaDevices.getUserMedia(constraints);
}

function stopCamera() {
  stream.getTracks().forEach((track) => {
    if (track.readyState === 'live' && track.kind === 'video') {
      track.stop();
    }
  });
}

/**
 * This method is used to render live camera tab.
 */
async function renderCamStream() {
  let inputBuffer;
  if (useWebnn) {
    inputBuffer = fastStyleTransferNet.preprocess(camElement);    
  } else {
    inputBuffer = getInputTensor(camElement, inputOptions);
    inputBuffer = tf.tensor(inputBuffer, inputOptions.inputDimensions, 'float32');
  }
  console.log('- Computing... ');
  const start = performance.now();
  let outputs;
  if (useWebnn) {
    outputs = await fastStyleTransferNet.compute(inputBuffer);
  } else {
    outputs = await fastStyleTransferNet.predict(inputBuffer);
  }
  computeTime = (performance.now() - start).toFixed(2);
  console.log(`  done in ${computeTime} ms.`);
  camElement.width = camElement.videoWidth;
  camElement.height = camElement.videoHeight;
  drawInput(camElement, 'camInCanvas');
  showPerfResult();
  await drawOutput(outputs, 'camInCanvas', 'camOutCanvas');
  if (!shouldStopFrame) {
    requestAnimationFrame(renderCamStream);
  }
}

function getMedianValue(array) {
  array = array.sort((a, b) => a - b);
  return array.length % 2 !== 0 ? array[Math.floor(array.length / 2)] :
      (array[array.length / 2 - 1] + array[array.length / 2]) / 2;
}

function drawInput(srcElement, canvasId) {
  const inputCanvas = document.getElementById(canvasId);
  const resizeRatio = Math.max(
      Math.max(srcElement.width / maxWidth, srcElement.height / maxHeight), 1);
  const scaledWidth = Math.floor(srcElement.width / resizeRatio);
  const scaledHeight = Math.floor(srcElement.height / resizeRatio);
  inputCanvas.height = scaledHeight;
  inputCanvas.width = scaledWidth;
  const ctx = inputCanvas.getContext('2d');
  ctx.drawImage(srcElement, 0, 0, scaledWidth, scaledHeight);
}

async function drawOutput(outputs, inCanvasId, outCanvasId) {
  let outputTensor, outputSize;
  if (useWebnn) {
    outputTensor = outputs.output.data;
    outputSize = outputs.output.dimensions;
  } else {
    const b = tf.transpose(outputs, [0, 3, 1, 2]);
    const buffer = await b.buffer();
    outputSize = b.shape;
    outputTensor = buffer.values;
  }
  const height = outputSize[2];
  const width = outputSize[3];
  const mean = [1, 1, 1, 1];
  const offset = [0, 0, 0, 0];
  const bytes = new Uint8ClampedArray(width * height * 4);
  const a = 255;

  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    const r = outputTensor[i] * mean[0] + offset[0];
    const g = outputTensor[i + height * width] * mean[1] + offset[1];
    const b = outputTensor[i + height * width * 2] * mean[2] + offset[2];
    bytes[j + 0] = Math.round(r);
    bytes[j + 1] = Math.round(g);
    bytes[j + 2] = Math.round(b);
    bytes[j + 3] = Math.round(a);
  }

  const imageData = new ImageData(bytes, width, height);
  const outCanvas = document.createElement('canvas');
  const outCtx = outCanvas.getContext('2d');
  outCanvas.width = width;
  outCanvas.height = height;
  outCtx.putImageData(imageData, 0, 0, 0, 0, outCanvas.width, outCanvas.height);

  const inputCanvas = document.getElementById(inCanvasId);
  const outputCanvas = document.getElementById(outCanvasId);
  outputCanvas.width = inputCanvas.width;
  outputCanvas.height = inputCanvas.height;
  const ctx = outputCanvas.getContext('2d');
  ctx.drawImage(outCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
}

function showPerfResult(timeInfo) {
  if (inputType === 'image') {
    $('#medianTime').html(`${timeInfo.medianTime} ms`);
    $('#averageTime').html(`${timeInfo.averageTime} ms`);
    $('#maxTime').html(`${timeInfo.maxTime} ms`);
    $('#minTime').html(`${timeInfo.minTime} ms`);
    $('#firstInferenceTime').html(`${timeInfo.firstInferenceTime} ms`);
  }
  $('#loadTime').html(`${loadTime} ms`);
  $('#buildTime').html(`${buildTime} ms`);
  
}

function addWarning(msg) {
  const div = document.createElement('div');
  div.setAttribute('class', 'alert alert-warning alert-dismissible fade show');
  div.setAttribute('role', 'alert');
  div.innerHTML = msg;
  const container = document.getElementById('container');
  container.insertBefore(div, container.childNodes[0]);
}

function populateTrendline(data) {
  const node = document.querySelector('#perf-trendline-container');
  const chartHeight = 150;
  const chartWidth = 300;
  node.querySelector('svg').setAttribute('width', chartWidth);
  node.querySelector('svg').setAttribute('height', chartHeight);

  const yMax = Math.max(...data);
  let yMin = 0;

  node.querySelector('.yMin').textContent = yMin + ' ms';
  node.querySelector('.yMax').textContent = yMax.toFixed(1) + ' ms';
  const xIncrement = chartWidth / (data.length - 1);

  node.querySelector('path')
    .setAttribute('d', `M${data.map((d, i) => `${i * xIncrement},${chartHeight - ((d - yMin) / (yMax - yMin)) * chartHeight}`).join('L')} `);
}

export function getInputTensor(inputElement, inputOptions) {
  const inputDimensions = inputOptions.inputDimensions;
  const tensor = new Float32Array(
      inputDimensions.slice(1).reduce((a, b) => a * b));

  inputElement.width = inputElement.videoWidth ||
      inputElement.naturalWidth;
  inputElement.height = inputElement.videoHeight ||
      inputElement.naturalHeight;

  let [channels, height, width] = inputDimensions.slice(1);
  const mean = inputOptions.mean || [0, 0, 0, 0];
  const std = inputOptions.std || [1, 1, 1, 1];
  const normlizationFlag = inputOptions.norm || false;
  const inputLayout = inputOptions.inputLayout;
  const imageChannels = 4; // RGBA

  if (inputLayout === 'nhwc') {
    [height, width, channels] = inputDimensions.slice(1);
  }
  const canvasElement = document.createElement('canvas');
  canvasElement.width = width;
  canvasElement.height = height;
  const canvasContext = canvasElement.getContext('2d');
  canvasContext.drawImage(inputElement, 0, 0, width, height);

  let pixels = canvasContext.getImageData(0, 0, width, height).data;

  if (normlizationFlag) {
    pixels = new Float32Array(pixels).map((p) => p / 255);
  }

  for (let c = 0; c < channels; ++c) {
    for (let h = 0; h < height; ++h) {
      for (let w = 0; w < width; ++w) {
        const value =
            pixels[h * width * imageChannels + w * imageChannels + c];
        if (inputLayout === 'nchw') {
          tensor[c * width * height + h * width + w] =
              (value - mean[c]) / std[c];
        } else {
          tensor[h * width * channels + w * channels + c] =
              (value - mean[c]) / std[c];
        }
      }
    }
  }
  return tensor;
}

export async function setBackend(backend, useWebnnFlag) {
  if (!backend) {
    // Use cpu by default for accuracy reason
    // See more details at:
    // https://github.com/webmachinelearning/webnn-polyfill/pull/32#issuecomment-763825323
    backend = 'cpu';
  }
  let tfjs;
  if (useWebnnFlag) {
    console.log('use webnn');
    useWebnn = useWebnnFlag;
    tfjs = navigator.ml.createContext().tf;
  } else {
    tfjs = tf;
  }
  const backends = ['webgl', 'webgpu', 'cpu'];
  if (!backends.includes(backend)) {
    if (backend) {
      console.warn(`${backend} backend is not supported.`);
    }
  } else {
    if (!(await tfjs.setBackend(backend))) {
      console.error(`Failed to set tf.js backend ${backend}.`);
    }
  }
  await tfjs.ready();
  console.info(
      `Uses tf.js ${tfjs.version_core}` +
      ` ${tfjs.getBackend()} backend.`);
}

export async function main() {
  try {
    let start;
    // Only do load() and build() when page first time loads and
    // there's new model choosed
    if (isFirstTimeLoad || isModelChanged) {
      if (fastStyleTransferNet !== undefined) {
        // Call dispose() to and avoid memory leak
        if (useWebnn) {
          fastStyleTransferNet.dispose();
        } else {
          // TODO: use tf.tidy() instead. https://js.tensorflow.org/api/latest/#tidy
          tf.dispose();
        }
      }
      if (useWebnn) {
        fastStyleTransferNet = new FastStyleTransferNet();
      }
      isFirstTimeLoad = false;
      isModelChanged = false;
      console.log(`- Model ID: ${modelId} -`);
      // UI shows model loading progress
      await showProgressComponent('current', 'pending', 'pending');
      console.log('- Loading weights... ');
      start = performance.now();
      let outputOperand;
      if (useWebnn) {
        console.log('use webnn')
        outputOperand = await fastStyleTransferNet.load(modelId);
      } else {
        fastStyleTransferNet = await tf.loadGraphModel(`./models/${modelId}/model.json`);
      }
      loadTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${loadTime} ms.`);
      // UI shows model building progress
      await showProgressComponent('done', 'current', 'pending');
      console.log('- Compiling... ');
      start = performance.now();
      if (useWebnn) {
        await fastStyleTransferNet.build(outputOperand);
      }
      buildTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${buildTime} ms.`);
    }
    // UI shows inferencing progress
    await showProgressComponent('done', 'done', 'current');
    if (inputType === 'image') {
      let inputBuffer;
      if (useWebnn) {
        inputBuffer = fastStyleTransferNet.preprocess(imgElement);    
      } else {
        inputBuffer = getInputTensor(imgElement, inputOptions);
        inputBuffer = tf.tensor(inputBuffer, inputOptions.inputDimensions, 'float32');
      }
      console.log('- Computing... ');
      start = performance.now();
      let outputs;
      console.time('First Inference Time');
      if (useWebnn) {
        outputs = await fastStyleTransferNet.compute(inputBuffer);
      } else {
        outputs = await fastStyleTransferNet.predict(inputBuffer);
      }
      console.timeEnd('First Inference Time');
      console.log('outputs: ', outputs);
      firstInferenceTime = (performance.now() - start).toFixed(2);
      console.log(`First inference time: ${firstInferenceTime} ms`);
      let times = [];
      let warmupTime = [];
      warmupTime.push(Number(firstInferenceTime));
      let numRuns = (new URLSearchParams(location.search)).get('numRuns');
      numRuns = numRuns === null ? 0 : parseInt(numRuns);
      if (numRuns > 0) {
        for (let i = 0; i < numRuns-1; i++) {
          start = performance.now();
          if (useWebnn) {
            await fastStyleTransferNet.compute(inputBuffer);
          } else {
            await fastStyleTransferNet.predict(inputBuffer);
          }
          let time = (performance.now() - start).toFixed(2);
          times.push(Number(time));
          console.log(`time ${i+1}: ${time} ms`);
        }
      } else {
        console.log('Start first 50 times compute...');
        $('#noticeInfo').html('Start first 50 times compute...');
        for (let i = 0; i < 50; i++) {
          start = performance.now();
          if (useWebnn) {
            await fastStyleTransferNet.compute(inputBuffer);
          } else {
            await fastStyleTransferNet.predict(inputBuffer);
          }
          let time = (performance.now() - start).toFixed(2);
          warmupTime.push(Number(time));
          console.log(`time ${i+1}: ${time} ms`);
          $('#noticeInfo').html(`Complete ${i+1} of first 50 times' predictions: ${time} ms`);
        }
        console.log('Done first 50 times compute...');
        console.log('Start next 101 times compute to get median inference time...');
        $('#noticeInfo').html('Start next 101 times compute...');
        console.log("Start at: ", (new Date()).toLocaleTimeString());
        for (let i = 0; i < 101; i++) {
          start = performance.now();
          if (useWebnn) {
            await fastStyleTransferNet.compute(inputBuffer);
          } else {
            await fastStyleTransferNet.predict(inputBuffer);
          }
          let time = (performance.now() - start).toFixed(2);
          times.push(Number(time));
          $('#noticeInfo').html(`Complete ${i+1} of last 101 times predictions: ${time} ms`);
          console.log(`101 predictions of time ${i+1}: ${time} ms`);
        }
        $('#noticeInfo').html('Done 101 times compute...');
        console.log("End at: ", (new Date()).toLocaleTimeString());
        populateTrendline(warmupTime.concat(times));
        console.log('Done next 101 times compute.');
        let timeString = "";
        for(let time of warmupTime.concat(times)) {
          timeString += time + ',';
        }
        console.log('all inference times:', timeString);
      }
      const averageTime = (times.reduce((acc, curr) => acc + curr, 0) / times.length).toFixed(2);
      console.log('averageTime: ', averageTime);
      const minTime = Math.min(...times);
      const maxTime = Math.max(...times);
      const medianTime = getMedianValue(times);
      const timeInfo = {
        loadTime,
        buildTime,
        firstInferenceTime,
        averageTime,
        minTime,
        maxTime,
        medianTime
      };
      console.log(timeInfo);
      await showProgressComponent('done', 'done', 'done');
      readyShowResultComponents();
      drawInput(imgElement, 'inputCanvas');
      await drawOutput(outputs, 'inputCanvas', 'outputCanvas');
      showPerfResult(timeInfo);
    } else if (inputType === 'camera') {
      await getMediaStream();
      camElement.srcObject = stream;
      shouldStopFrame = false;
      camElement.onloadedmediadata = await renderCamStream();
      await showProgressComponent('done', 'done', 'done');
      readyShowResultComponents();
    } else {
      throw Error(`Unknown inputType ${inputType}`);
    }
    $('#gallery .gallery-image').on('click', handleImageSwitch);
  } catch (error) {
    console.log(error);
    addWarning(error.message);
  }
}
