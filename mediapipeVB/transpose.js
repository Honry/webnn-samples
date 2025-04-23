import {Tensor} from './tensor.js';

export function transpose(input, {permutation} = {}) {
  const inpPermutation = permutation ??
        new Array(input.rank).fill(0).map((e, i, a) => a.length - i - 1);

  const outputShape = new Array(input.rank).fill(0).map(
      (e, i, a) => input.shape[inpPermutation[i]]);
  const output = new Tensor(outputShape);
  for (let inputIndex = 0; inputIndex < input.size; ++inputIndex) {
    const inputValue = input.getValueByIndex(inputIndex);
    const inputLocation = input.locationFromIndex(inputIndex);
    const outputLocation = new Array(output.rank);
    for (let i = 0; i < inpPermutation.length; ++i) {
      outputLocation[i] = inputLocation[inpPermutation[i]];
    }
    output.setValueByLocation(outputLocation, inputValue);
  }
  return output;
}