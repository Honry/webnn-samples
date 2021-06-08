'use strict';

const maxWidth = 380;
const maxHeight = 380;
const imgElement = document.getElementById('feedElement');
imgElement.src = './images/content-images/travelspace.jpg';
let modelId = 'starry-night';
let isFirstTimeLoad = true;
let isModelChanged = false;
let fastStyleTransferNet;
let loadTime = 0;
const inputDimensions = [1, 540, 540, 3];

$(document).ready(() => {
  $('.icdisplay').hide();
  $('.badge').html(modelId);
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
    $('#feedElement').removeAttr('height');
    $('#feedElement').removeAttr('width');
    imgElement.src = URL.createObjectURL(files[0]);
  }
});

// Click handler to do inference with switched <img> element
$('#gallery .gallery-image').on('click', async (e) => {
  const newModelId = $(e.target).attr('id');
  if (newModelId !== modelId) {
    isModelChanged = true;
    modelId = newModelId;
    const modelName = $(`#${modelId}`).attr('title');
    $('.badge').html(modelName);
    $('#gallery .gallery-item').removeClass('hl');
    $(e.target).parent().addClass('hl');
  }
});

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
  const buffer = await outputs.buffer();
  const outputTensor = buffer.values;
  const outputSize = outputs.shape;
  const height = outputSize[1];
  const width = outputSize[2];
  const mean = [1, 1, 1, 1];
  const offset = [0, 0, 0, 0];
  const bytes = new Uint8ClampedArray(width * height * 4);
  const a = 255;

  for (let i = 0; i < height * width; ++i) {
    let j = i * 4;
    let r = outputTensor[i * 3] * mean[0] + offset[0];
    let g = outputTensor[i * 3 + 1] * mean[1] + offset[1];
    let b = outputTensor[i * 3 + 2] * mean[2] + offset[2];
    bytes[j + 0] = Math.round(r);
    bytes[j + 1] = Math.round(g);
    bytes[j + 2] = Math.round(b);
    bytes[j + 3] = Math.round(a);
  }
  const imageData = new ImageData(bytes, width, height);
  const outCanvas = document.createElement('canvas');
  let outCtx = outCanvas.getContext('2d');
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
  $('.icdisplay').show();
  $('.shoulddisplay').show();
  $('#medianTime').html(`${timeInfo.medianTime} ms`);
  $('#averageTime').html(`${timeInfo.averageTime} ms`);
  $('#maxTime').html(`${timeInfo.maxTime} ms`);
  $('#minTime').html(`${timeInfo.minTime} ms`);
  $('#firstInferenceTime').html(`${timeInfo.firstInferenceTime} ms`);
  $('#loadTime').html(`${loadTime} ms`);
  
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

export function getInputTensor(inputElement) {
  const tensor = new Float32Array(
      inputDimensions.slice(1).reduce((a, b) => a * b));

  inputElement.width = inputElement.videoWidth ||
      inputElement.naturalWidth;
  inputElement.height = inputElement.videoHeight ||
      inputElement.naturalHeight;

  let [height, width, channels] = inputDimensions.slice(1);
  const mean = [0, 0, 0, 0];
  const std = [1, 1, 1, 1];
  const imageChannels = 4; // RGBA

  const canvasElement = document.createElement('canvas');
  canvasElement.width = width;
  canvasElement.height = height;
  const canvasContext = canvasElement.getContext('2d');
  canvasContext.drawImage(inputElement, 0, 0, width, height);

  let pixels = canvasContext.getImageData(0, 0, width, height).data;

  for (let c = 0; c < channels; ++c) {
    for (let h = 0; h < height; ++h) {
      for (let w = 0; w < width; ++w) {
        const value =
            pixels[h * width * imageChannels + w * imageChannels + c];
        tensor[h * width * channels + w * channels + c] =
            (value - mean[c]) / std[c];
      }
    }
  }
  return tensor;
}

export async function setBackend(backend) {
  if (!backend) {
    backend = 'webgl';
  }
  const backends = ['webgl', 'webgpu'];
  if (!backends.includes(backend)) {
    console.warn(`${backend} backend is not supported.`);
  } else {
    if (!(await tf.setBackend(backend))) {
      console.error(`Failed to set tf.js backend ${backend}.`);
    }
  }
  await tf.ready();
  $('#noticeInfo').html(`Uses tf.js ${tf.version_core} ${tf.getBackend()} backend.`);
  console.info(`Uses tf.js ${tf.version_core} ${tf.getBackend()} backend.`);
}

/**
 * Downloads the values from the `tensorContainer` from any `tf.Tensor`s found
 * within the `tensorContainer`. Returns a promise of `TypedArray` or
 * `TypedArray[]` that resolves when the computation has finished.
 *
 * The values are asynchronously downloaded in parallel.
 *
 * @param tensorContainer The container of tensors to be downloaded.
 */
async function downloadValuesFromTensorContainer(tensorContainer) {
  let valueContainer;
  if (tensorContainer instanceof tf.Tensor) {
    valueContainer = await tensorContainer.data();
  } else if (Array.isArray(tensorContainer)) {
    // Start value downloads from all tensors.
    const valuePromiseContainer = tensorContainer.map(async item => {
      if (item instanceof tf.Tensor) {
        return item.data();
      }
      return item;
    });
    // Wait until all values are downloaded.
    valueContainer = await Promise.all(valuePromiseContainer);
  } else if (tensorContainer != null && typeof tensorContainer === 'object') {
    const valuePromiseContainer = [];
    // Start value downloads from all tensors.
    for (const property in tensorContainer) {
      if (tensorContainer[property] instanceof tf.Tensor) {
        valuePromiseContainer.push(tensorContainer[property].data());
      } else {
        valuePromiseContainer.push(tensorContainer[property]);
      }
    }
    // Wait until all values are downloaded.
    valueContainer = await Promise.all(valuePromiseContainer);
  }
  return valueContainer;
}

function appendRow(tbody, ...cells) {
  const tr = document.createElement('tr');
  cells.forEach(c => {
    const td = document.createElement('td');
    if (c instanceof HTMLElement) {
      td.appendChild(c);
    } else {
      td.innerHTML = c;
    }
    tr.appendChild(td);
  });
  tbody.appendChild(tr);
}

function showKernelTime(profileInfo) {
  const aggregatedTbody = document.querySelector('#kernels-aggregated tbody');
  const individualTbody = document.querySelector('#kernels-individual tbody');
  profileInfo.aggregatedKernels.forEach(r => {
    const nameSpan = document.createElement('span');
    nameSpan.setAttribute('title', r.name);
    nameSpan.textContent = r.name;
    appendRow(aggregatedTbody, nameSpan, r.timeMs.toFixed(2));
  });
  profileInfo.kernels.forEach(kernel => {
    const nameSpan = document.createElement('span');
    nameSpan.setAttribute('title', kernel.name);
    nameSpan.textContent = kernel.name;
    let inputInfo;
    kernel.inputShapes.forEach((inputShape, index) => {
      if (inputInfo == null) {
        inputInfo = '';
      } else {
        inputInfo += '<br>';
      }
      if (inputShape == null) {
        inputInfo += `input${index}: null`;
      } else {
        inputInfo += `input${index}: ${inputShape.length}D[${inputShape}]`;
      }
    });
    appendRow(individualTbody, nameSpan, kernel.kernelTimeMs.toFixed(2), inputInfo, kernel.outputShapes, kernel.extraInfo);
  });
}

// Predict to execute for profiling memory usage
async function profileInference(model, inputTensor) {
  
  const kernelInfo = await tf.profile(async () => {
    let start = performance.now();
    const res = model.execute(inputTensor);
    await downloadValuesFromTensorContainer(res);
    const profileTime = (performance.now() - start).toFixed(2);
    console.log('predict time in profiling: ', profileTime);
    tf.dispose(res);
  });


  kernelInfo.kernels =
      kernelInfo.kernels.sort((a, b) => b.kernelTimeMs - a.kernelTimeMs);
  kernelInfo.aggregatedKernels = aggregateKernelTime(kernelInfo.kernels);
  return kernelInfo;
}

/**
 * Aggregate kernels by name and sort the array in non-ascending order of time.
 * Return an array of objects with `name` and `timeMs` fields.
 *
 * @param {Array<Object>} kernels An array of kernel information objects. Each
 *     object must include `name` (string) and `kernelTimeMs` (number) fields.
 */
function aggregateKernelTime(kernels) {
  const aggregatedKernelTime = {};
  kernels.forEach(kernel => {
    const oldAggregatedKernelTime = aggregatedKernelTime[kernel.name];
    if (oldAggregatedKernelTime == null) {
      aggregatedKernelTime[kernel.name] = kernel.kernelTimeMs;
    } else {
      aggregatedKernelTime[kernel.name] =
          oldAggregatedKernelTime + kernel.kernelTimeMs;
    }
  });

  return Object.entries(aggregatedKernelTime)
      .map(([name, timeMs]) => ({name, timeMs}))
      .sort((a, b) => b.timeMs - a.timeMs);
}

export async function main() {
  try {
    $('.icdisplay').hide();
    $('.shoulddisplay').hide();
    let start;
    // Only do load() when page first time loads and
    // there's new model choosed
    if (isFirstTimeLoad || isModelChanged) {
      isFirstTimeLoad = false;
      isModelChanged = false;
      console.log(`- Model ID: ${modelId} -`);
      $('#noticeInfo').html(`Loading model: ${modelId}...`);
      start = performance.now();
      fastStyleTransferNet = await tf.loadGraphModel(`./models/${modelId}/model.json`);
      loadTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${loadTime} ms.`);
    }
    const inputBuffer = getInputTensor(imgElement);
    const inputTensor = tf.tensor(inputBuffer, inputDimensions, 'float32');
    console.log('- Predict... ');
    $('#noticeInfo').html(`Start first inference...`);
    start = performance.now();
    console.time('First Inference Time');
    const outputs = fastStyleTransferNet.execute(inputTensor);
    const value = await downloadValuesFromTensorContainer(outputs);
    console.timeEnd('First Inference Time');
    const firstInferenceTime = (performance.now() - start).toFixed(2);
    console.log(`First inference time: ${firstInferenceTime} ms`);
    let times = [];
    let warmupTime = [];
    warmupTime.push(Number(firstInferenceTime));
    console.log('Start first 50 times compute...');
    $('#noticeInfo').html('Start first 50 times compute...');
    for (let i = 0; i < 50; i++) {
      start = performance.now();
      const outputs = fastStyleTransferNet.execute(inputTensor);
      const value = await downloadValuesFromTensorContainer(outputs);
      let time = (performance.now() - start).toFixed(2);
      tf.dispose(outputs);
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
      const outputs = fastStyleTransferNet.execute(inputTensor);
      const value = await downloadValuesFromTensorContainer(outputs);
      let time = (performance.now() - start).toFixed(2);
      tf.dispose(outputs);
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
    const profileInfo = await profileInference(fastStyleTransferNet, inputTensor);
    console.log('profileInfo: ', profileInfo);
    showKernelTime(profileInfo);
    const averageTime = (times.reduce((acc, curr) => acc + curr, 0) / times.length).toFixed(2);
    const minTime = Math.min(...times);
    const maxTime = Math.max(...times);
    const medianTime = getMedianValue(times);
    const timeInfo = {
      loadTime,
      firstInferenceTime,
      averageTime,
      minTime,
      maxTime,
      medianTime
    };
    drawInput(imgElement, 'inputCanvas');
    await drawOutput(outputs, 'inputCanvas', 'outputCanvas');
    showPerfResult(timeInfo);
  } catch (error) {
    console.log(error);
    addWarning(error.message);
  }
}
