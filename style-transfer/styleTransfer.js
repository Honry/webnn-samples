
'use strict';

function sizeOfShape(shape) {
  return shape.reduce((a, b) => {
    return a * b;
  });
}

async function buildConstantByNpy(builder, url, reshape=false) {
  const dataTypeMap = new Map([
    ['f2', { type: 'float16', array: Uint16Array }],
    ['f4', { type: 'float32', array: Float32Array }],
    ['f8', { type: 'float64', array: Float64Array }],
    ['i1', { type: 'int8', array: Int8Array }],
    ['i2', { type: 'int16', array: Int16Array }],
    ['i4', { type: 'int32', array: Int32Array }],
    ['i8', { type: 'int64', array: BigInt64Array }],
    ['u1', { type: 'uint8', array: Uint8Array }],
    ['u2', { type: 'uint16', array: Uint16Array }],
    ['u4', { type: 'uint32', array: Uint32Array }],
    ['u8', { type: 'uint64', array: BigUint64Array }],
  ]);
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();
  const npArray = new numpy.Array(new Uint8Array(buffer));
  if (!dataTypeMap.has(npArray.dataType)) {
    throw new Error(`Data type ${npArray.dataType} is not supported.`);
  }
  let dimensions = npArray.shape;
  const type = dataTypeMap.get(npArray.dataType).type;
  const TypedArrayConstructor = dataTypeMap.get(npArray.dataType).array;
  const typedArray = new TypedArrayConstructor(sizeOfShape(dimensions));
  const dataView = new DataView(npArray.data.buffer);
  const littleEndian = npArray.byteOrder === '<';
  for (let i = 0; i < sizeOfShape(dimensions); ++i) {
    typedArray[i] = dataView[`get` + type[0].toUpperCase() + type.substr(1)](
      i * TypedArrayConstructor.BYTES_PER_ELEMENT, littleEndian);
  }
  // Onnx's conv2d = webnn's add(conv2d + bias)
  // Workaround: reshape bias input from onnx's conv2d op in order to broadcast
  // output of onnx's conv2d op with bias input
  if (reshape) {
    if (dimensions.length !== 1) {
      throw new Error(`Unable to reshape conv's bias with dimensions ${dimensions}`);
    }
    dimensions = [1, dimensions[0], 1, 1];
  }
  return builder.constant({ type, dimensions }, typedArray);
}

/* eslint max-len: ["error", { "code": 130 }] */

// Style Transfer Baseline Model
export class StyleTransfer {
  constructor() {
    this.model_ = null;
    this.compiledModel_ = null;
    this.dimensions_ = [1, 3, 224, 224];
  }

  async load(baseUrl) {
    const nn = navigator.ml.getNeuralNetworkContext();
    const builder = nn.createModelBuilder();

    // Wanming
    // Create constants by loading pre-trained data from .npy files.
    const weightConv1 = await buildConstantByNpy(builder, baseUrl + 'conv1_conv2d_weight.npy');
    const biasConv1 = await buildConstantByNpy(builder, baseUrl + 'conv1_conv2d_bias.npy', true);
    const weightIn1 = await buildConstantByNpy(builder, baseUrl + 'in1_weight.npy');
    const biasIn1 = await buildConstantByNpy(builder, baseUrl + 'in1_bias.npy');
    const weightConv2 = await buildConstantByNpy(builder, baseUrl + 'conv2_conv2d_weight.npy');
    const biasConv2 = await buildConstantByNpy(builder, baseUrl + 'conv2_conv2d_bias.npy', true);
    const weightIn2 = await buildConstantByNpy(builder, baseUrl + 'in2_weight.npy');
    const biasIn2 = await buildConstantByNpy(builder, baseUrl + 'in2_bias.npy');
    const weightConv3 = await buildConstantByNpy(builder, baseUrl + 'conv3_conv2d_weight.npy');
    const biasConv3 = await buildConstantByNpy(builder, baseUrl + 'conv3_conv2d_bias.npy', true);
    const weightIn3 = await buildConstantByNpy(builder, baseUrl + 'in3_weight.npy');
    const biasIn3 = await buildConstantByNpy(builder, baseUrl + 'in3_bias.npy');

    const weightRes1Conv1 = await buildConstantByNpy(builder, baseUrl + 'res1_conv1_conv2d_weight.npy');
    const biasRes1Conv1 = await buildConstantByNpy(builder, baseUrl + 'res1_conv1_conv2d_bias.npy', true);
    const weightRes1In1 = await buildConstantByNpy(builder, baseUrl + 'res1_in1_weight.npy');
    const biasRes1In1 = await buildConstantByNpy(builder, baseUrl + 'res1_in1_bias.npy');
    const weightRes1Conv2 = await buildConstantByNpy(builder, baseUrl + 'res1_conv2_conv2d_weight.npy');
    const biasRes1Conv2 = await buildConstantByNpy(builder, baseUrl + 'res1_conv2_conv2d_bias.npy', true);
    const weightRes1In2 = await buildConstantByNpy(builder, baseUrl + 'res1_in2_weight.npy');
    const biasRes1In2 = await buildConstantByNpy(builder, baseUrl + 'res1_in2_bias.npy');

    const weightRes2Conv1 = await buildConstantByNpy(builder, baseUrl + 'res2_conv1_conv2d_weight.npy');
    const biasRes2Conv1 = await buildConstantByNpy(builder, baseUrl + 'res2_conv1_conv2d_bias.npy', true);
    const weightRes2In1 = await buildConstantByNpy(builder, baseUrl + 'res2_in1_weight.npy');
    const biasRes2In1 = await buildConstantByNpy(builder, baseUrl + 'res2_in1_bias.npy');
    const weightRes2Conv2 = await buildConstantByNpy(builder, baseUrl + 'res2_conv2_conv2d_weight.npy');
    const biasRes2Conv2 = await buildConstantByNpy(builder, baseUrl + 'res2_conv2_conv2d_bias.npy', true);
    const weightRes2In2 = await buildConstantByNpy(builder, baseUrl + 'res2_in2_weight.npy');
    const biasRes2In2 = await buildConstantByNpy(builder, baseUrl + 'res2_in2_bias.npy');

    const weightRes3Conv1 = await buildConstantByNpy(builder, baseUrl + 'res3_conv1_conv2d_weight.npy');
    const biasRes3Conv1 = await buildConstantByNpy(builder, baseUrl + 'res3_conv1_conv2d_bias.npy', true);
    const weightRes3In1 = await buildConstantByNpy(builder, baseUrl + 'res3_in1_weight.npy');
    const biasRes3In1 = await buildConstantByNpy(builder, baseUrl + 'res3_in1_bias.npy');
    const weightRes3Conv2 = await buildConstantByNpy(builder, baseUrl + 'res3_conv2_conv2d_weight.npy');
    const biasRes3Conv2 = await buildConstantByNpy(builder, baseUrl + 'res3_conv2_conv2d_bias.npy', true);
    const weightRes3In2 = await buildConstantByNpy(builder, baseUrl + 'res3_in2_weight.npy');
    const biasRes3In2 = await buildConstantByNpy(builder, baseUrl + 'res3_in2_bias.npy');

    const weightRes4Conv1 = await buildConstantByNpy(builder, baseUrl + 'res4_conv1_conv2d_weight.npy');
    const biasRes4Conv1 = await buildConstantByNpy(builder, baseUrl + 'res4_conv1_conv2d_bias.npy', true);
    const weightRes4In1 = await buildConstantByNpy(builder, baseUrl + 'res4_in1_weight.npy');
    const biasRes4In1 = await buildConstantByNpy(builder, baseUrl + 'res4_in1_bias.npy');
    const weightRes4Conv2 = await buildConstantByNpy(builder, baseUrl + 'res4_conv2_conv2d_weight.npy');
    const biasRes4Conv2 = await buildConstantByNpy(builder, baseUrl + 'res4_conv2_conv2d_bias.npy', true);
    const weightRes4In2 = await buildConstantByNpy(builder, baseUrl + 'res4_in2_weight.npy');
    const biasRes4In2 = await buildConstantByNpy(builder, baseUrl + 'res4_in2_bias.npy');

    const weightRes5Conv1 = await buildConstantByNpy(builder, baseUrl + 'res5_conv1_conv2d_weight.npy');
    const biasRes5Conv1 = await buildConstantByNpy(builder, baseUrl + 'res5_conv1_conv2d_bias.npy', true);
    const weightRes5In1 = await buildConstantByNpy(builder, baseUrl + 'res5_in1_weight.npy');
    const biasRes5In1 = await buildConstantByNpy(builder, baseUrl + 'res5_in1_bias.npy');
    const weightRes5Conv2 = await buildConstantByNpy(builder, baseUrl + 'res5_conv2_conv2d_weight.npy');
    const biasRes5Conv2 = await buildConstantByNpy(builder, baseUrl + 'res5_conv2_conv2d_bias.npy', true);
    const weightRes5In2 = await buildConstantByNpy(builder, baseUrl + 'res5_in2_weight.npy');
    const biasRes5In2 = await buildConstantByNpy(builder, baseUrl + 'res5_in2_bias.npy');

    const weightDecConv1 = await buildConstantByNpy(builder, baseUrl + 'deconv1_conv2d_weight.npy');
    const biasDecConv1 = await buildConstantByNpy(builder, baseUrl + 'deconv1_conv2d_bias.npy', true);
    const weightIn4 = await buildConstantByNpy(builder, baseUrl + 'in4_weight.npy');
    const biasIn4 = await buildConstantByNpy(builder, baseUrl + 'in4_bias.npy');

    const weightDecConv2 = await buildConstantByNpy(builder, baseUrl + 'deconv2_conv2d_weight.npy');
    const biasDecConv2 = await buildConstantByNpy(builder, baseUrl + 'deconv2_conv2d_bias.npy', true);
    const weightIn5 = await buildConstantByNpy(builder, baseUrl + 'in5_weight.npy');
    const biasIn5 = await buildConstantByNpy(builder, baseUrl + 'in5_bias.npy');

    const weightDecConv3 = await buildConstantByNpy(builder, baseUrl + 'deconv3_conv2d_weight.npy');
    const biasDecConv3 = await buildConstantByNpy(builder, baseUrl + 'deconv3_conv2d_bias.npy', true);
    const padding1 = builder.constant(
      { type: 'int32', dimensions: [4, 2] }, new Int32Array([0, 0, 0, 0, 1, 1, 1, 1]));
    const padding4 = builder.constant(
      { type: 'int32', dimensions: [4, 2] }, new Int32Array([0, 0, 0, 0, 4, 4, 4, 4]));
    // Build up the network.
    const input = builder.input('input', { type: 'float32', dimensions: this.dimensions_ });
    const pad63 = builder.pad(input, padding4, { mode: "reflection" });
    const conv64 = builder.add(builder.conv2d(pad63, weightConv1), biasConv1);
    const instanceN11n65 = builder.instanceNormalization(conv64, { scale: weightIn1, bias: biasIn1 });
    const relu66 = builder.relu(instanceN11n65);
    const pad67 = builder.pad(relu66, padding1, { mode: "reflection" });
    const conv68 = builder.add(builder.conv2d(pad67, weightConv2), biasConv2);
    const instanceN11n69 = builder.instanceNormalization(conv68, { scale: weightIn2, bias: biasIn2 });
    const relu70 = builder.relu(instanceN11n69);
    const pad71 = builder.pad(relu70, padding1, { mode: "reflection" });
    const conv72 = builder.add(builder.conv2d(pad71, weightConv3), biasConv3);
    const instanceN11n73 = builder.instanceNormalization(conv72, { scale: weightIn3, bias: biasIn3 });
    const relu74 = builder.relu(instanceN11n73);
    const pad75 = builder.pad(relu74, padding1, { mode: "reflection" });
    const conv76 = builder.add(builder.conv2d(pad75, weightRes1Conv1), biasRes1Conv1);
    const instanceN11n77 = builder.instanceNormalization(conv76, { scale: weightRes1In1, bias: biasRes1In1 });
    const relu78 = builder.relu(instanceN11n77);
    const pad79 = builder.pad(relu78, padding1, { mode: "reflection" });
    const conv80 = builder.add(builder.conv2d(pad79, weightRes1Conv2), biasRes1Conv2);
    const instanceN11n81 = builder.instanceNormalization(conv80, { scale: weightRes1In2, bias: biasRes1In2 });

    const add82 = builder.add(relu74, instanceN11n81);
    const pad83 = builder.pad(add82, padding1, { mode: "reflection" });
    const conv84 = builder.add(builder.conv2d(pad83, weightRes2Conv1), biasRes2Conv1);
    const instanceN11n85 = builder.instanceNormalization(conv84, { scale: weightRes2In1, bias: biasRes2In1 });
    const relu86 = builder.relu(instanceN11n85); // [1, 128, 224, 224]
    const pad87 = builder.pad(relu86, padding1, { mode: "reflection" });
    const conv88 = builder.add(builder.conv2d(pad87, weightRes2Conv2), biasRes2Conv2);
    const instanceN11n89 = builder.instanceNormalization(conv88, { scale: weightRes2In2, bias: biasRes2In2 });
    const add90 = builder.add(add82, instanceN11n89);

    const pad91 = builder.pad(add90, padding1, { mode: "reflection" });
    const conv92 = builder.add(builder.conv2d(pad91, weightRes3Conv1), biasRes3Conv1);
    const instanceN11n93 = builder.instanceNormalization(conv92, { scale: weightRes3In1, bias: biasRes3In1 });
    const relu94 = builder.relu(instanceN11n93);
    const pad95 = builder.pad(relu94, padding1, { mode: "reflection" });
    const conv96 = builder.add(builder.conv2d(pad95, weightRes3Conv2), biasRes3Conv2);
    const instanceN11n97 = builder.instanceNormalization(conv96, { scale: weightRes3In2, bias: biasRes3In2 });
    const add98 = builder.add(add90, instanceN11n97);

    const pad99 = builder.pad(add98, padding1, { mode: "reflection" });
    const conv100 = builder.add(builder.conv2d(pad99, weightRes4Conv1), biasRes4Conv1);
    const instanceN11n101 = builder.instanceNormalization(conv100, { scale: weightRes4In1, bias: biasRes4In1 });
    const relu102 = builder.relu(instanceN11n101);
    const pad103 = builder.pad(relu102, padding1, { mode: "reflection" });
    const conv104 = builder.add(builder.conv2d(pad103, weightRes4Conv2), biasRes4Conv2);
    const instanceN11n105 = builder.instanceNormalization(conv104, { scale: weightRes4In2, bias: biasRes4In2 });
    const add106 = builder.add(add98, instanceN11n105);

    const pad107 = builder.pad(add106, padding1, { mode: "reflection" });
    const conv108 = builder.add(builder.conv2d(pad107, weightRes5Conv1), biasRes5Conv1);
    const instanceN11n109 = builder.instanceNormalization(conv108, { scale: weightRes5In1, bias: biasRes5In1 });
    const relu110 = builder.relu(instanceN11n109);
    const pad111 = builder.pad(relu110, padding1, { mode: "reflection" });
    const conv112 = builder.add(builder.conv2d(pad111, weightRes5Conv2), biasRes5Conv2);
    const instanceN11n113 = builder.instanceNormalization(conv112, { scale: weightRes5In2, bias: biasRes5In2 });
    const add114 = builder.add(add106, instanceN11n113);

    const shape116 = [1, 2, 4, 8]; // TBD
    const upsample139 = builder.resample(add114, { scales: [1.0, 1.0, 2.0, 2.0] }); // TODO: incase the shape has None value

    const pad140 = builder.pad(upsample139, padding1, { mode: "reflection" });
    const conv141 = builder.add(builder.conv2d(pad140, weightDecConv1), biasDecConv1);
    const instanceN11n142 = builder.instanceNormalization(conv141, { scale: weightIn4, bias: biasIn4 });

    const relu143 = builder.relu(instanceN11n142);

    const shape145 = [2, 4, 6, 8] // TBD
    const upsample168 = builder.resample(relu143, { scales: [1.0, 1.0, 2.0, 2.0] }); // TODO: incase the shape has None value

    const pad169 = builder.pad(upsample168, padding1, { mode: "reflection" });
    const conv170 = builder.add(builder.conv2d(pad169, weightDecConv2), biasDecConv2);
    const instanceN11n171 = builder.instanceNormalization(conv170, { scale: weightIn5, bias: biasIn5 });
    const relu172 = builder.relu(instanceN11n171);
    const pad173 = builder.pad(relu172, padding4, { mode: "reflection" });
    const output = builder.add(builder.conv2d(pad173, weightDecConv3), biasDecConv3);
    this.model_ = builder.createModel({ 'output': output });
    // Wanming - end
  }

  async compile(options) {
    this.compiledModel_ = await this.model_.compile(options);
  }

  async compute(inputBuffer) {
    const inputs = { input: { buffer: inputBuffer } };
    return await this.compiledModel_.compute(inputs);
  }
}
