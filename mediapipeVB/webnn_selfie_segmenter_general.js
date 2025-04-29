"use strict";

// Selfie-Segmenter WebNN model
export class WebnnSelfieSegmenterGeneral {
  constructor(deviceType) {
    this.deviceType_ = deviceType;
    this.context_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.inputTensor_ = null;
    this.outputTensor_ = null;
    this.outputShape_ = [1, 256, 256, 1];
  }

  async buildConv_(
    input,
    index,
    activation = "",
    options = {}
  ) {
    const weightInfo = this.weightsInfo_[`conv${index}`];
    const weightBuffer = this.weightsBuffer_.slice(
      weightInfo.dataOffset,
      weightInfo.dataOffset + weightInfo.byteLength
    );

    const weights = this.builder_.constant(
      { shape: weightInfo.shape, dataType: "float32" },
      new Float32Array(weightBuffer)
    );

    const biasInfo = this.biasesInfo_[`conv${index}`];
    const biasBuffer = this.biasesBuffer_.slice(
      biasInfo.dataOffset,
      biasInfo.dataOffset + biasInfo.byteLength
    );
    options.bias = this.builder_.constant(
      { shape: biasInfo.shape, dataType: "float32" },
      new Float32Array(biasBuffer)
    );

    if (this.layout === "nhwc") {
      const isDepthwise = options.groups > 1 && options.groups == input["shape"][3];
      options.filterLayout = isDepthwise ? "ihwo" : "ohwi";
      options.inputLayout = this.layout;
    }
    const conv2d = this.builder_.conv2d(input, weights, options);

    if (activation === "relu") {
      return this.builder_.relu(conv2d);
    } else if (activation === "sigmoid") {
      return this.builder_.sigmoid(conv2d);
    } else {
      return conv2d;
    }
  }

  // Subgraph 0:
  // input -> Conv -> Add (B: addB_) -> Clip (min: 0, max: 6) -> Mul (A: mulA_) -> Mul (A: Conv)
  //           |                                                                    ^
  //           v                                                                    |
  //           ----------------------------------------------------------------------
  async buildSubGraph0_(input, convIndex, convOptions = {}) {
    const conv = await this.buildConv_(input, convIndex, "", convOptions);
    const add = this.builder_.add(conv, this.addB_);
    const clip = this.builder_.clamp(add, { minValue: 0, maxValue: 6 });
    const mul = this.builder_.mul(this.mulA_, clip);
    return this.builder_.mul(conv, mul);
  }

  // Subgraph 1: (if optionInput presents, it will be used as input for Mul)
  // input -> AveragePool -> Conv -> Relu -> Conv -> Sigmoid -> Mul
  //  |                                                          ^
  //  v                                                          |
  //  ----------------------------------------------------------(or optionInput)
  async buildSubGraph1_(input, convIndex, stride, optionInput = undefined) {
    const strides = [stride, stride]; // AveragePool has the same kernel_shape and strides
    const avgPool2d = this.builder_.averagePool2d(await input, {
      windowDimensions: strides,
      strides,
      layout: this.layout,
    });
    // convIndex
    const conv1 = await this.buildConv_(avgPool2d, convIndex, "relu");
    // convIndex + 1
    const conv2 = await this.buildConv_(conv1, convIndex + 1, "sigmoid");
    if (optionInput) {
      return this.builder_.mul(optionInput, conv2);
    } else {
      return this.builder_.mul(input, conv2);
    }
  }

  async load() {
    this.context_ = await navigator.ml.createContext({
      deviceType: this.deviceType_,
    });

    // Choose the layout based on the preferred input layout of the context.
    this.layout = this.context_.opSupportLimits().preferredInputLayout;
    this.inputShape =
      this.layout === "nhwc" ? [1, 256, 256, 3] : [1, 3, 256, 256];

    // Load the weights, bias and info files.
    const weightsResponse = await fetch(
      `./weights/general/weights_${this.layout}.bin`
    );
    this.weightsBuffer_ = await weightsResponse.arrayBuffer();

    const weightInfoResponse = await fetch(
      `./weights/general/weights_${this.layout}.json`
    );
    this.weightsInfo_ = await weightInfoResponse.json();

    // Different layouts have the same bias
    const biasResponse = await fetch(`./weights/general/biases.bin`);
    this.biasesBuffer_ = await biasResponse.arrayBuffer();
    this.biasInfoResponse_ = await fetch(`./weights/general/biases.json`);
    this.biasesInfo_ = await this.biasInfoResponse_.json();

    this.builder_ = new MLGraphBuilder(this.context_);
    const strides = [2, 2];

    const inputDesc = {
      dataType: "float32",
      shape: this.inputShape,
    };
    const input = this.builder_.input("input", inputDesc);
    inputDesc.writable = true;
    this.inputTensor_ = await this.context_.createTensor(inputDesc);
    this.outputTensor_ = await this.context_.createTensor({
      dataType: "float32",
      shape: this.outputShape_,
      readable: true,
    });

    this.addB_ = this.builder_.constant(
      { dataType: "float32", shape: [1, 1, 1, 1] },
      new Float32Array([3])
    );
    this.mulA_ = this.builder_.constant(
      { dataType: "float32", shape: [] },
      new Float32Array([0.1666666716337204])
    );

    // name: mul_1 (contains conv0) Conv__221
    const subGraph0_0 = await this.buildSubGraph0_(input, 0, {
      strides,
      padding: [0, 1, 0, 1],
    });

    // Conv__224
    const conv1 = await this.buildConv_(subGraph0_0, 1, "relu");
    // Conv__225
    const conv2 = await this.buildConv_(
      conv1,
      2,
      "relu",
      {
        strides,
        padding: [0, 1, 0, 1],
        groups: 16,
      }
    );

    // name: multiply, Conv__228, Conv__229 (contains conv3, conv4)
    // AvgeragePool2d: strides: [64, 64]
    const subGraph1_0 = await this.buildSubGraph1_(conv2, 3, 64);

    // name: Conv__230
    const conv5 = await this.buildConv_(subGraph1_0, 5, "");
    const conv6 = await this.buildConv_(conv5, 6, "relu");
    // name: Conv__235
    const conv7 = await this.buildConv_(
      conv6,
      7,
      "relu",
      {
        strides,
        padding: [0, 1, 0, 1],
        groups: 72,
      }
    );
    // name: Conv__236
    const conv8 = await this.buildConv_(conv7, 8, "");
    // name: Conv__239
    const conv9 = await this.buildConv_(conv8, 9, "relu");
    // name: Conv__240
    const conv10 = await this.buildConv_(
      conv9,
      10,
      "relu",
      {
        padding: [1, 1, 1, 1],
        groups: 88,
      }
    );
    // name: Conv__241
    const conv11 = await this.buildConv_(conv10, 11, "");

    // name: add__xeno_compat__1
    const add0 = this.builder_.add(conv11, conv8);

    // Conv__246
    const subGraph0_1 = await this.buildSubGraph0_(add0, 12);

    // Conv__249
    const subGraph0_2 = await this.buildSubGraph0_(
      subGraph0_1,
      13,
      {
        strides,
        padding: [1, 2, 1, 2],
        groups: 96,
      }
    );

    // Conv__252, Conv__253 (contains: conv14, conv15)
    // averagePool2d: strides: [16, 16]
    const subGraph1_1 = await this.buildSubGraph1_(subGraph0_2, 14, 16);

    // Conv__254
    const conv15 = await this.buildConv_(subGraph1_1, 16, "");

    // Conv__257
    const subGraph0_3 = await this.buildSubGraph0_(conv15, 17);

    // Conv__260
    const subGraph0_4 = await this.buildSubGraph0_(
      subGraph0_3,
      18,
      {
        padding: [2, 2, 2, 2],
        groups: 128,
      }
    );

    // Conv__263, Conv264 (contains: conv19, conv20)
    // averagePool2d: strides: [16, 16]
    const subGraph1_2 = await this.buildSubGraph1_(subGraph0_4, 19, 16);

    // Conv__265
    const conv21 = await this.buildConv_(subGraph1_2, 21, "");

    // name: add_1__xeno_compat__1
    const add1 = this.builder_.add(conv21, conv15);

    // Conv__268
    const subGraph0_5 = await this.buildSubGraph0_(add1, 22);

    // Conv__271
    const subGraph0_6 = await this.buildSubGraph0_(
      subGraph0_5,
      23,
      {
        padding: [2, 2, 2, 2],
        groups: 128,
      }
    );

    // Conv__274, Conv__275 (contains: conv24, conv25)
    // averagePool2d: strides: [16, 16]
    const subGraph1_3 = await this.buildSubGraph1_(subGraph0_6, 24, 16);

    // Conv__276
    const conv26 = await this.buildConv_(subGraph1_3, 26, "");

    // name: add_2__xeno_compat__1
    const add2 = this.builder_.add(conv26, add1);

    // Conv__279
    const subGraph0_7 = await this.buildSubGraph0_(add2, 27);
    // Conv__282
    const subGraph0_8 = await this.buildSubGraph0_(
      subGraph0_7,
      28,
      {
        padding: [2, 2, 2, 2],
        groups: 96,
      }
    );

    // Conv__285, Conv__286 (contains: conv29, conv30)
    // averagePool2d: strides: [16, 16]
    const subGraph1_4 = await this.buildSubGraph1_(subGraph0_8, 29, 16);

    // Conv__287
    const conv31 = await this.buildConv_(subGraph1_4, 31, "");

    // name: add_3__xeno_compat__1
    const add3 = this.builder_.add(conv31, add2);

    // Conv__290
    const subGraph0_9 = await this.buildSubGraph0_(add3, 32);
    // Conv__293
    const subGraph0_10 = await this.buildSubGraph0_(
      subGraph0_9,
      33,
      {
        padding: [2, 2, 2, 2],
        groups: 96,
      }
    );

    // Conv__296, Conv__297 (contains: conv34, conv35)
    // averagePool2d: strides: [16, 16]
    const subGraph1_5 = await this.buildSubGraph1_(subGraph0_10, 34, 16);

    // Conv__298
    const conv36 = await this.buildConv_(subGraph1_5, 36, "");

    // name: add_4__xeno_compat__1
    const add4 = this.builder_.add(conv36, add3);

    // Conv__300
    const conv37 = await this.buildConv_(add4, 37, "relu");

    // name: average_pooling2d_6_avgpool/AvgPool
    // averagePool2d: strides: [16, 16]
    const avgPool2d0 = this.builder_.averagePool2d(add4, {
      windowDimensions: [16, 16],
      strides: [16, 16],
      layout: this.layout,
    });

    // Conv__299
    const conv38 = await this.buildConv_(avgPool2d0, 38, "sigmoid");

    // name: multiply_6
    const mul0 = this.builder_.mul(conv37, conv38);

    // Resize__153
    const resample0 = this.builder_.resample2d(mul0, {
      sizes: [32, 32],
      mode: "linear",
      axes: this.layout === "nhwc" ? [1, 2] : [2, 3],
    });

    // Conv__301
    const conv39 = await this.buildConv_(resample0, 39, "");

    // name: add_5__xeno_compat__1
    const add5 = this.builder_.add(conv39, add0);

    // Conv__314, Conv__315 (contains: conv40, conv41)
    // averagePool2d: strides: [32, 32]
    const subGraph1_6 = await this.buildSubGraph1_(add5, 40, 32, add0);

    // name: add_6__xeno_compat__1
    const add6 = this.builder_.add(subGraph1_6, conv39);

    // Conv__316
    const conv42 = await this.buildConv_(add6, 42, "relu");
    // Conv__319
    const conv43 = await this.buildConv_(
      conv42,
      43,
      "relu",
      {
        padding: [1, 1, 1, 1],
        groups: 24,
      }
    );

    // name: add_7__xeno_compat__1
    const add7 = this.builder_.add(conv42, conv43);

    // Resize__178
    const resample1 = this.builder_.resample2d(add7, {
      sizes: [64, 64],
      mode: "linear",
      axes: this.layout === "nhwc" ? [1, 2] : [2, 3],
    });

    // Conv__320
    const conv44 = await this.buildConv_(resample1, 44, "");

    // name: add_8__xeno_compat__1
    const add8 = this.builder_.add(conv5, conv44);

    // Conv__333, Conv__334 (contains: conv45, conv46)
    // AveragePool2d: strides: [64, 64]
    const subGraph1_7 = await this.buildSubGraph1_(add8, 45, 64, conv5);

    // name: add_9__xeno_compat__1
    const add9 = this.builder_.add(subGraph1_7, conv44);

    // Conv__335
    const conv47 = await this.buildConv_(add9, 47, "relu");
    // Conv__338
    const conv48 = await this.buildConv_(
      conv47,
      48,
      "relu",
      {
        padding: [1, 1, 1, 1],
        groups: 16,
      }
    );

    // name: add_10__xeno_compat__1
    const add10 = this.builder_.add(conv47, conv48);

    // Resize__203
    const resample2 = this.builder_.resample2d(add10, {
      sizes: [128, 128],
      mode: "linear",
      axes: this.layout === "nhwc" ? [1, 2] : [2, 3],
    });

    // Conv__339
    const conv49 = await this.buildConv_(resample2, 49, "");

    // name: add_11__xeno_compat__1
    const add11 = this.builder_.add(subGraph0_0, conv49);

    // Conv__352, Conv__353 (contains: conv50, conv51)
    // AveragePool2d: strides: [128, 128]
    const subGraph1_8 = await this.buildSubGraph1_(add11, 50, 128, subGraph0_0);

    // name: add_12__xeno_compat__1
    const add12 = this.builder_.add(subGraph1_8, conv49);

    // Conv__354
    const conv52 = await this.buildConv_(add12, 52, "relu");
    // Conv__357
    const conv53 = await this.buildConv_(
      conv52,
      53,
      "relu",
      {
        padding: [1, 1, 1, 1],
        groups: 16,
      }
    );

    // name: add_13__xeno_compat__1
    const add13 = this.builder_.add(conv52, conv53);

    // ConvTranspose
    const convTransposeWInfo = this.weightsInfo_["convTranspose0"];
    const convTransposeWBuffer = this.weightsBuffer_.slice(
      convTransposeWInfo.dataOffset,
      convTransposeWInfo.dataOffset + convTransposeWInfo.byteLength
    );
    const convTransposeW = this.builder_.constant(
      { shape: convTransposeWInfo.shape, dataType: "float32" },
      new Float32Array(convTransposeWBuffer)
    );
    const convTransposeB = this.builder_.constant(
      { dataType: "float32", shape: [1] },
      new Float32Array([0.53271484375])
    );
    const convTranspose = this.builder_.convTranspose2d(add13, convTransposeW, {
      bias: convTransposeB,
      padding: this.layout === "nhwc" ? [0, 0, 0, 0] : [0, 1, 0, 1],
      strides: [2, 2],
      outputSizes: [256, 256],
      filterLayout: this.layout === "nhwc" ? "ohwi" : "iohw",
      inputLayout: this.layout,
    });

    // name: activation_10
    const sigmoid = this.builder_.sigmoid(convTranspose);
    if (this.layout === "nhwc") {
      return sigmoid;
    } else {
      return this.builder_.reshape(sigmoid, this.outputShape_);
    }
  }

  async build(outputOperand) {
    this.graph_ = await this.builder_.build({ segment_back: outputOperand });
  }

  async compute(inputBuffer) {
    this.context_.writeTensor(this.inputTensor_, inputBuffer);
    const inputs = { input: this.inputTensor_ };
    const outputs = { segment_back: this.outputTensor_ };
    this.context_.dispatch(this.graph_, inputs, outputs);
    const results = await this.context_.readTensor(this.outputTensor_);
    return new Float32Array(results);
  }
}
