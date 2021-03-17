import { StyleTransfer } from './styleTransfer.js';

export class TransferStyle {
  constructor(inputElement) {
    this.styleTransfer_ = new StyleTransfer();
    this.logger_ = null;
    this.inputElement_ = inputElement;
    this.inputSize_ = [1, 3, 224, 224];
  }

  log_(message, sep = false, append = true) {
    if (this.logger) {
      this.logger.innerHTML = (append ? this.logger.innerHTML : '') + message +
        (sep ? '<br>' : '');
    }
  }

  async prepare() {
    console.log('prepare')
    this.log_(' - Loading weights... ');
    let start = performance.now();
    await this.styleTransfer_.load('./weights/');
    const modelLoadTime = (performance.now() - start).toFixed(2);
    this.log_(`done in <span class='text-primary'>` +
      `${modelLoadTime}</span> ms.`, true);
    this.log_(' - Compiling... ');
    console.log('compile...')
    start = performance.now();
    await this.styleTransfer_.compile();
    const modelCompileTime = (performance.now() - start).toFixed(2);
    this.log_(`done in <span class='text-primary'>` +
      `${modelCompileTime}</span> ms.`, true);
    console.log('complie done');
  }

  async process() {
    console.log('process')
    const inputData = this.getInputTensor_();
    this.log_(' - Computing... ');
    let start = performance.now();
    const outputs = await this.styleTransfer_.compute(inputData);
    console.log('compute done');
    console.log('outputs: ', outputs)
    const computeTime = (performance.now() - start).toFixed(2);
    this.log_(`done in <span class='text-primary'>` +
      `${computeTime}</span> ms.`, true);
    return outputs;
  }

  getInputTensor_() {
    let tensor = new Float32Array(this.inputSize_.slice(1).reduce((a, b) => a * b));

    this.inputElement_.width = this.inputElement_.videoWidth || this.inputElement_.naturalWidth;
    this.inputElement_.height = this.inputElement_.videoHeight || this.inputElement_.naturalHeight;

    const [channels, height, width] = this.inputSize_.slice(1);
    const mean = [0, 0, 0, 0];
    const std = [1, 1, 1, 1];
    const imageChannels = 4; // RGBA

    let canvasElement = document.createElement('canvas');
    canvasElement.width = width;
    canvasElement.height = height;
    let canvasContext = canvasElement.getContext('2d');
    canvasContext.drawImage(this.inputElement_, 0, 0, width, height);

    let pixels = canvasContext.getImageData(0, 0, width, height).data;

    for (let c = 0; c < channels; ++c) {
      for (let h = 0; h < height; ++h) {
        for (let w = 0; w < width; ++w) {
          let value = pixels[h * width * imageChannels + w * imageChannels + c];
          tensor[c * width * height + h * width + w] = (value - mean[c]) / std[c];
        }
      }
    }
    return tensor;
  }
}
