'use strict';

import {DeepLabV3MNV2TFLite} from './deeplabv3_mnv2_tflite.js';
import {DeepLabV3MNV2ONNX} from './deeplabv3_mnv2_onnx.js';
import {SelfieSegmentation} from './seflie_segmentation_tflite.js';
import {DeepLabV3MNV2Nchw} from './deeplabv3_mnv2_nchw.js';
import {DeepLabV3MNV2Nhwc} from './deeplabv3_mnv2_nhwc.js';
import {showProgressComponent, readyShowResultComponents} from '../common/ui.js';
import {getInputTensor, getMedianValue, getDevicePreference, sizeOfShape} from '../common/utils.js';
import {buildWebGL2Pipeline} from './lib/webgl2/webgl2Pipeline.js';

const imgElement = document.getElementById('feedElement');
imgElement.src = './images/test.jpg';
const camElement = document.getElementById('feedMediaElement');
const outputCanvas = document.getElementById('outputCanvas');
let instanceType = 'deeplabnchw';
let rafReq;
let isFirstTimeLoad = true;
let inputType = 'image';
let netInstance = null;
let labels = null;
let stream = null;
let loadTime = 0;
let buildTime = 0;
let computeTime = 0;
let inputOptions;
let outputBuffer;
let model;
let modelChanged = false;
let backgroundImageSource = document.getElementById('00-img');
let backgroundType = 'img'; // 'none', 'blur', 'image'

$(document).ready(() => {
  $('.icdisplay').hide();
});

$(window).on('load', () => {
});

$('#modelBtns .btn').on('change', async (e) => {
  modelChanged = true;
  instanceType = $(e.target).attr('id');
  if (inputType === 'camera') cancelAnimationFrame(rafReq);
  await main();
});

// Click trigger to do inference with <img> element
$('#img').click(async () => {
  if (inputType === 'camera') cancelAnimationFrame(rafReq);
  if (stream !== null) {
    stopCamera();
  }
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
  if (!isFirstTimeLoad) {
    await main();
  }
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
  await drawOutput(outputBuffer, imgElement);
});

async function fetchLabels(url) {
  const response = await fetch(url);
  const data = await response.text();
  return data.split('\n');
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
  const inputBuffer = getInputTensor(camElement, inputOptions);
  console.log('- Computing... ');
  const start = performance.now();
  if (instanceType === 'deeplabnchw' || instanceType === 'deeplabnhwc') {
    netInstance.compute(inputBuffer, outputBuffer);
  } else if (instanceType === 'deeplabonnx') {
    outputBuffer = await netInstance.compute(model, inputBuffer);
  } else {
    outputBuffer = netInstance.compute(model, inputBuffer);
  }
  computeTime = (performance.now() - start).toFixed(2);
  console.log(`  done in ${computeTime} ms.`);
  showPerfResult();
  await drawOutput(outputBuffer, camElement);
  rafReq = requestAnimationFrame(renderCamStream);
}

async function drawOutput(outputBuffer, srcElement) {

  if (instanceType.startsWith('deeplab')) {
    outputBuffer = tf.tidy(() => {
      const a = tf.tensor(outputBuffer, netInstance.outputDimensions, 'float32');
      let axis = 3;
      if (instanceType === 'deeplabnchw') {
        axis = 1;
      }
      const b = tf.argMax(a, axis);
      const c = tf.tensor(b.dataSync(), b.shape, 'float32');
      return c.dataSync();
    });
  }
  console.log('output: ', outputBuffer);
  outputCanvas.width = srcElement.naturalWidth | srcElement.videoWidth;
  outputCanvas.height = srcElement.naturalHeight | srcElement.videoHeight;
  const pipeline = buildWebGL2Pipeline(
    srcElement,
    backgroundImageSource,
    backgroundType,
    inputOptions.inputResolution,
    outputCanvas,
    outputBuffer,
  );
  const postProcessingConfig = {
    smoothSegmentationMask: true,
    jointBilateralFilter: {sigmaSpace: 1, sigmaColor: 0.1},
    coverage: [0.5, 0.75],
    lightWrapping: 0.3,
    blendMode: 'screen',
  }
  pipeline.updatePostProcessingConfig(postProcessingConfig);
  await pipeline.render();
}

function showPerfResult(medianComputeTime = undefined) {
  $('#loadTime').html(`${loadTime} ms`);
  $('#buildTime').html(`${buildTime} ms`);
  if (medianComputeTime !== undefined) {
    $('#computeLabel').html('Median inference time:');
    $('#computeTime').html(`${medianComputeTime} ms`);
  } else {
    $('#computeLabel').html('Inference time:');
    $('#computeTime').html(`${computeTime} ms`);
  }
}

function constructNetObject(type) {
  const netObject = {
    'deeplabnchw': new DeepLabV3MNV2Nchw(),
    'deeplabnhwc': new DeepLabV3MNV2Nhwc(),
    'deeplabtflite': new DeepLabV3MNV2TFLite(),
    'deeplabonnx': new DeepLabV3MNV2ONNX(),
    'sstflite': new SelfieSegmentation(),
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
    let start;
    // Set 'numRuns' param to run inference multiple times
    const params = new URLSearchParams(location.search);
    let numRuns = params.get('numRuns');
    numRuns = numRuns === null ? 1 : parseInt(numRuns);

    if (numRuns < 1) {
      addWarning('The value of param numRuns must be greater than or equal' +
          ' to 1.');
      return;
    }
    // Only do load() and build() when model first time loads and
    // there's new model choosed
    if (isFirstTimeLoad || modelChanged) {
      modelChanged = false;
      netInstance = constructNetObject(instanceType);
      inputOptions = netInstance.inputOptions;
      labels = await fetchLabels(inputOptions.labelUrl);
      isFirstTimeLoad = false;
      console.log(`- Model: ${instanceType}-`);
      // UI shows model loading progress
      await showProgressComponent('current', 'pending', 'pending');
      console.log('- Loading model... ');
      start = performance.now();
      const devicePreference = getDevicePreference();
      model = await netInstance.load(devicePreference);
      loadTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${loadTime} ms.`);
      // UI shows model building progress
      await showProgressComponent('done', 'current', 'pending');
      if (instanceType === 'deeplabnchw' || instanceType === 'deeplabnhwc') {
        console.log('- Building... ');
        start = performance.now();
        netInstance.build(model);
        buildTime = (performance.now() - start).toFixed(2);
        console.log(`  done in ${buildTime} ms.`);
      }
    }
    // UI shows inferencing progress
    await showProgressComponent('done', 'done', 'current');
    outputBuffer = new Float32Array(sizeOfShape(netInstance.outputDimensions));
    if (inputType === 'image') {
      const inputBuffer = getInputTensor(imgElement, inputOptions);
      console.log('- Computing... ');
      const computeTimeArray = [];
      let medianComputeTime;
      for (let i = 0; i < numRuns; i++) {
        start = performance.now();
        if (instanceType === 'deeplabnchw' || instanceType === 'deeplabnhwc') {
          netInstance.compute(inputBuffer, outputBuffer);
        } else if (instanceType === 'deeplabonnx') {
          outputBuffer = await netInstance.compute(model, inputBuffer);
        } else {
          outputBuffer = netInstance.compute(model, inputBuffer);
        }
        computeTime = (performance.now() - start).toFixed(2);
        console.log(`  compute time ${i+1}: ${computeTime} ms`);
        computeTimeArray.push(Number(computeTime));
      }
      if (numRuns > 1) {
        medianComputeTime = getMedianValue(computeTimeArray);
        medianComputeTime = medianComputeTime.toFixed(2);
        console.log(`  median compute time: ${medianComputeTime} ms`);
      }

      await showProgressComponent('done', 'done', 'done');
      readyShowResultComponents();
      await drawOutput(outputBuffer, imgElement);
      showPerfResult(medianComputeTime);
    } else if (inputType === 'camera') {
      await getMediaStream();
      camElement.srcObject = stream;
      camElement.onloadedmediadata = await renderCamStream();
      await showProgressComponent('done', 'done', 'done');
      readyShowResultComponents();
    } else {
      throw Error(`Unknown inputType ${inputType}`);
    }
  } catch (error) {
    console.log(error);
    addWarning(error.message);
  }
}
