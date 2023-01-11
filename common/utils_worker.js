'use strict';

importScripts('../common/libs/numpy_worker.js');

function sizeOfShape(shape) {
  return shape.reduce((a, b) => {
    return a * b;
  });
}

// This function is used for reading buffer from a given url,
// which will be exported to node.js environment as well,
// so we use 'fs' module for examples ran in node.js and
// fetch() method for examples ran in browser.
async function getBufferFromUrl(url) {
  let arrayBuffer;
  if (globalThis.fetch) {
    const response = await fetch(url);
    arrayBuffer = await response.arrayBuffer();
  } else {
    const fs = await import('fs');
    const uint8Array = await fs.promises.readFile(url);
    arrayBuffer = uint8Array.buffer;
  }
  return arrayBuffer;
}

async function buildConstantByNpy(builder, url) {
  const dataTypeMap = new Map([
    ['f2', {type: 'float16', array: Uint16Array}],
    ['f4', {type: 'float32', array: Float32Array}],
    ['f8', {type: 'float64', array: Float64Array}],
    ['i1', {type: 'int8', array: Int8Array}],
    ['i2', {type: 'int16', array: Int16Array}],
    ['i4', {type: 'int32', array: Int32Array}],
    ['i8', {type: 'int64', array: BigInt64Array}],
    ['u1', {type: 'uint8', array: Uint8Array}],
    ['u2', {type: 'uint16', array: Uint16Array}],
    ['u4', {type: 'uint32', array: Uint32Array}],
    ['u8', {type: 'uint64', array: BigUint64Array}],
  ]);
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();
  const npArray = new numpy.Array(new Uint8Array(buffer));
  if (!dataTypeMap.has(npArray.dataType)) {
    throw new Error(`Data type ${npArray.dataType} is not supported.`);
  }
  const dimensions = npArray.shape;
  const type = dataTypeMap.get(npArray.dataType).type;
  const TypedArrayConstructor = dataTypeMap.get(npArray.dataType).array;
  const typedArray = new TypedArrayConstructor(sizeOfShape(dimensions));
  const dataView = new DataView(npArray.data.buffer);
  const littleEndian = npArray.byteOrder === '<';
  for (let i = 0; i < sizeOfShape(dimensions); ++i) {
    typedArray[i] = dataView[`get` + type[0].toUpperCase() + type.substr(1)](
        i * TypedArrayConstructor.BYTES_PER_ELEMENT, littleEndian);
  }
  return builder.constant({type, dimensions}, typedArray);
}

// Get median value from an array of Number
function getMedianValue(array) {
  array = array.sort((a, b) => a - b);
  return array.length % 2 !== 0 ? array[Math.floor(array.length / 2)] :
      (array[array.length / 2 - 1] + array[array.length / 2]) / 2;
}

// Set tf.js backend based WebNN's 'MLDeviceType' option
async function setPolyfillBackend(device) {
  // Simulate WebNN's device selection using various tf.js backends.
  // MLDeviceType: ['default', 'gpu', 'cpu']
  // 'default' or 'gpu': tfjs-backend-webgl, 'cpu': tfjs-backend-wasm
  if (!device) device = 'gpu';
  // Use 'webgl' by default for better performance.
  // Note: 'wasm' backend may run failed on some samples since
  // some ops aren't supported on 'wasm' backend at present
  const backend = device === 'cpu' ? 'wasm' : 'webgl';
  const tf = navigator.ml.createContext().tf;
  if (tf) {
    if (!(await tf.setBackend(backend))) {
      throw new Error(`Failed to set tf.js backend ${backend}.`);
    }
    await tf.ready();
    let backendInfo = backend == 'wasm' ? 'WASM' : 'WebGL';
    if (backendInfo == 'WASM') {
      const hasSimd = tf.env().features['WASM_HAS_SIMD_SUPPORT'];
      const hasThreads = tf.env().features['WASM_HAS_MULTITHREAD_SUPPORT'];
      if (hasThreads && hasSimd) {
        backendInfo += ' (SIMD + threads)';
      } else if (hasThreads && !hasSimd) {
        backendInfo += ' (threads)';
      } else if (!hasThreads && hasSimd) {
        backendInfo += ' (SIMD)';
      }
    }
  }
}