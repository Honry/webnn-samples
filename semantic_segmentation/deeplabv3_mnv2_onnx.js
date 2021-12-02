'use strict';

/* eslint max-len: ["error", {"code": 120}] */

// DeepLab V3 MobileNet V2 model with onnxruntime web
export class DeepLabV3MNV2ONNX {
  constructor() {
    this.inputOptions = {
        mean: [127.5, 127.5, 127.5],
        std: [127.5, 127.5, 127.5],
        scaledFlag: false,
        inputLayout: 'nhwc',
        labelUrl: './labels/labels.txt',
        inputDimensions: [1, 321, 321, 3], // deeplab
        inputResolution: [321, 321],
      };
      this.outputDimensions = [1,321,321,21];
  }

  async load() {
    // override path of wasm files - for each file
    ort.env.wasm.wasmPaths = {
      'ort-wasm.wasm': location.href + '/onnxruntime-web/js/web/dist/ort-wasm.wasm',
      'ort-wasm-simd.wasm': location.href + '/onnxruntime-web/js/web/dist/ort-wasm-simd.wasm',
      'ort-wasm-threaded.wasm': location.href + '/onnxruntime-web/js/web/dist/ort-wasm-threaded.wasm',
      'ort-wasm-simd-threaded.wasm': location.href + '/onnxruntime-web/js/web/dist/ort-wasm-simd-threaded.wasm'
    };
    ort.env.wasm.numThreads = 1;
    // Create the model runner with the model.
    const model = './models/deeplab_mobilenetv2_321_no_argmax.onnx';
    const session = ort.InferenceSession.create(model, {
        executionProviders: ['wasm', {name: 'webnn', devicePreference: 1}], logSeverityLevel: 0 });
    return session;
  }

  async compute(session, inputData) {
    const inputTensor = new ort.Tensor('float32', inputData, this.inputOptions.inputDimensions);
    // prepare feeds. use model input names as keys.
    const feeds = {'sub_2': inputTensor};
    // feed inputs and run
    const results = await session.run(feeds);

    // read from results
    const outputBuffer = results.ResizeBilinear_2.data;
    return outputBuffer;
  }
}
