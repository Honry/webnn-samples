'use strict';

import { TransferStyle } from './transferStyle.js';

const maxWidth = 380;
const maxHeight = 380;
const imgElement = document.getElementById('feedElement');
imgElement.src = './images/content-images/amber.jpg';
const camElement = document.getElementById('feedMediaElement');
let reqId = 0;

// Click trigger to do inference with <img> element
$('#img').click(async () => {
  drawInput(imgElement, 'inputCanvas');
  if (reqId !== 0) {
    cancelAnimationFrame(reqId);
    reqId = 0;
  }
});

// Click trigger to do inference with switched <img> element
$("#gallery .gallery-image").click((e) => {
  const modelId = $(e.target).attr('id');
  $("#gallery .gallery-item").removeClass('hl');
  $(e.target).parent().addClass('hl');
});

let inputFileElement = document.getElementById('input');
inputFileElement.addEventListener('change', (e) => {
  let files = e.target.files;
  if (files.length > 0) {
    imgElement.src = URL.createObjectURL(files[0]);
  }
}, false);

$('#feedElement').on('load', async () => {
  drawInput(imgElement, 'inputCanvas');
  await drawOutput(imgElement, 'inputCanvas', 'outputCanvas');
 });

// Click trigger to do inference with <video> media element
$('#cam').click(async () => {
  console.log('camera...');
  const stream = await getMediaStream();
  camElement.srcObject = stream;
  camElement.onloadedmediadata = await renderCamStream();
});

async function getMediaStream() {
  // Support 'user' facing mode at first
  let constraints = { audio: false, video: { facingMode: 'user' } };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  return stream;
};

/**
 * This method is used to render camera video.
 */
async function renderCamStream() {
  camElement.width = camElement.videoWidth;
  camElement.height = camElement.videoHeight;
  drawInput(camElement, 'camInCanvas');
  // await drawOutput(camElement, 'camInCanvas', 'camOutCanvas');
  reqId = requestAnimationFrame(renderCamStream);
}

function drawInput(srcElement, canvasId) {
  console.log('drawimg')
  const inputCanvas = document.getElementById(canvasId);
  const resizeRatio = Math.max(Math.max(srcElement.width / maxWidth, srcElement.height / maxHeight), 1);
  const scaledWidth = Math.floor(srcElement.width / resizeRatio);
  const scaledHeight = Math.floor(srcElement.height / resizeRatio);
  inputCanvas.height = scaledHeight;
  inputCanvas.width = scaledWidth;
  const ctx = inputCanvas.getContext('2d');
  ctx.drawImage(srcElement, 0, 0, scaledWidth, scaledHeight);
};

async function drawOutput(srcElement, inCanvasId, outCanvasId) {
  const transferStyle = new TransferStyle(srcElement);
  await transferStyle.prepare();
  const outputs = await transferStyle.process();
  const outputTensor = outputs.output.buffer;
  const outputSize = outputs.output.dimensions;
  console.log('outputSize: ', outputSize);
  const height = outputSize[2];
  const width = outputSize[3];
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
};

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
  } catch (error) {
    console.log(error);
    addWarning(error.message);
  }
}
