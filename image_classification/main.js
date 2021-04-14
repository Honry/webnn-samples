'use strict';

import {MobileNetNchw} from './mobilenet_nchw.js';
import {MobileNetNhwc} from './mobilenet_nhwc.js';
import {SqueezeNetNchw} from './squeezenet_nchw.js';
import {SqueezeNetNhwc} from './squeezenet_nhwc.js';
import {showProgressComponent, readyShowResultComponents} from '../common/ui.js';
import {getInputTensor} from '../common/utils.js';

const maxWidth = 380;
const maxHeight = 380;
const imgElement = document.getElementById('feedElement');
imgElement.src = './images/test.jpg';
const camElement = document.getElementById('feedMediaElement');
let modelName;
let shouldStopFrame = false;
let inputType = 'image';
let netInstance = null;
let labels = null;
let stream = null;
let loadTime = 0;
let buildTime = 0;
let computeTime = 0;
let layout = 'nchw';
const nchwOptions = {
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
  norm: true,
  nchwFlag: true,
  labelUrl: './labels/labels1000.txt',
  inputDimensions: [1, 3, 224, 224],
};
const nhwcOptions = {
  mean: [127.5, 127.5, 127.5],
  std: [127.5, 127.5, 127.5],
  labelUrl: './labels/labels1001.txt',
  inputDimensions: [1, 224, 224, 3],
};
let layoutOptions = nchwOptions;

async function fetchLabels(url) {
  const response = await fetch(url);
  const data = await response.text();
  return data.split('\n');
}

$(document).ready(() => {
  $('.icdisplay').hide();
});

$('#modelBtns .btn').on('change', async (e) => {
    modelName = $(e.target).attr('id');
    await main();
});

$('#layoutBtns .btn').on('change', async (e) => {
  layout = $(e.target).attr('id');
  if (layout === 'nchw') {
    layoutOptions = nchwOptions;
  } else {
    layoutOptions = nhwcOptions;
  }
  await main();
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
  const inputBuffer = getInputTensor(camElement, layoutOptions);
  console.log('- Computing... ');
  const start = performance.now();
  const outputs = await netInstance.compute(inputBuffer);
  computeTime = (performance.now() - start).toFixed(2);
  console.log(`  done in ${computeTime} ms.`);
  camElement.width = camElement.videoWidth;
  camElement.height = camElement.videoHeight;
  drawInput(camElement, 'camInCanvas');
  showPerfResult();
  await drawOutput(outputs, labels);
  if (!shouldStopFrame) {
    requestAnimationFrame(renderCamStream);
  }
}

// Get top 3 classes of labels from output tensor
function getTopClasses(tensor, labels) {
  const probs = Array.from(tensor);
  const indexes = probs.map((prob, index) => [prob, index]);
  let sorted = indexes.sort((a, b) => {
    if (a[0] === b[0]) {
      return 0;
    }
    return a[0] < b[0] ? -1 : 1;
  });
  sorted.reverse();
  const classes = [];

  for (let i = 0; i < 3; ++i) {
    const prob = sorted[i][0];
    let index = sorted[i][1];
    let c = {
      label: labels[index],
      prob: (prob * 100).toFixed(2)
    }
    classes.push(c);
  }

  return classes;
};

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

async function drawOutput(outputs, labels) {
  const outputTensor = outputs.output.data;
  const labelClasses = getTopClasses(outputTensor, labels);

  $('#inferenceresult').show();
  labelClasses.forEach((c, i) => {
    console.log(`\tlabel: ${c.label}, probability: ${c.prob}%`);
    let labelElement = document.getElementById(`label${i}`);
    let probElement = document.getElementById(`prob${i}`);
    labelElement.innerHTML = `${c.label}`;
    probElement.innerHTML = `${c.prob}%`;
  });
}

function showPerfResult() {
  $('#loadTime').html(`${loadTime} ms`);
  $('#buildTime').html(`${buildTime} ms`);
  $('#computeTime').html(`${computeTime} ms`);
}

function constructNetObject(type) {
  const netObject = {
    'mobilenetnchw': new MobileNetNchw(),
    'mobilenetnhwc': new MobileNetNhwc(),
    'squeezenetnchw': new SqueezeNetNchw(),
    'squeezenetnhwc': new SqueezeNetNhwc(),
  };

  return netObject[type];
}

function addWarning(msg) {
  const div = document.createElement('div');
  div.setAttribute('class', 'alert alert-warning alert-dismissible fade show');
  div.setAttribute('role', 'alert');
  div.innerHTML = msg;
  const container = document.getElementById('container');
  container.insertBefore(div, container.childNodes[0]);
}

export async function main() {
  try {
    if (modelName !== undefined) {
      $('#nomodelInfo').hide();
      $('input[type="radio"]').attr('disabled', true);
      labels = await fetchLabels(layoutOptions.labelUrl);
      if (netInstance !== null) {
        // Call dispose() to and avoid memory leak
        netInstance.dispose();
      }
      netInstance = constructNetObject(modelName + layout);
      console.log(`- Model name: ${modelName}, Model layout: ${layout} -`);
      // UI shows model loading progress
      await showProgressComponent('current', 'pending', 'pending');
      console.log('- Loading weights... ');
      let start = performance.now();
      const outputOperand = await netInstance.load();
      loadTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${loadTime} ms.`);
      // UI shows model building progress
      await showProgressComponent('done', 'current', 'pending');
      console.log('- Building... ');
      start = performance.now();
      await netInstance.build(outputOperand);
      buildTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${buildTime} ms.`);
      // UI shows inferencing progress
      await showProgressComponent('done', 'done', 'current');
      if (inputType === 'image') {
        const inputBuffer = getInputTensor(imgElement, layoutOptions);
        console.log('- Computing... ');
        start = performance.now();
        const outputs = await netInstance.compute(inputBuffer);
        computeTime = (performance.now() - start).toFixed(2);
        console.log('output: ', outputs);
        console.log(`  done in ${computeTime} ms.`);
        await showProgressComponent('done', 'done', 'done');
        readyShowResultComponents();
        drawInput(imgElement, 'inputCanvas');
        await drawOutput(outputs, labels);
        showPerfResult();
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
      $('input[type="radio"]').attr('disabled', false);
  }
  } catch (error) {
    console.log(error);
    addWarning(error.message);
  }
}
