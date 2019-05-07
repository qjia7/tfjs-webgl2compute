/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// The differences with webgpu backend:
// Add 'computeWorkGroupSize' to choose a best configuration.

import {util} from '@tensorflow/tfjs-core';

const arrayProduct = (arr: number[]) => {
  if (!arr.length) {
    throw new Error('Cannot compute product of empty array.');
  }
  let product = 1;
  for (let i = 0; i < arr.length; i++) {
    product *= arr[i];
  }
  return product;
};

// Computes dispatch geometry based on layout of output dimensions and
// workGroupSize.
export function computeDispatch(
    layout: {x: number[], y: number[], z: number[]}, outputShape: number[],
    workGroupSize: [number, number, number] = [1, 1, 1],
    elementsPerThread: [number, number, number] =
        [1, 1, 1]): [number, number, number] {
  return [
    Math.ceil(
        arrayProduct(layout.x.map(d => outputShape[d])) /
        (workGroupSize[0] * elementsPerThread[0])),
    Math.ceil(
        arrayProduct(layout.y.map(d => outputShape[d])) /
        (workGroupSize[1] * elementsPerThread[1])),
    Math.ceil(
        arrayProduct(layout.z.map(d => outputShape[d])) /
        (workGroupSize[2] * elementsPerThread[2]))
  ];
}

// TODO: Figure out what kind of workgroup size is best?
export function computeWorkGroupSize(outputShape: number[]):
    [number, number, number] {
  const size = util.sizeFromShape(outputShape);
  let x = 16;
  if (size > 512) {
    x = 512;
  } else if (size > 256) {
    x = 256;
  } else if (size > 128) {
    x = 128;
  } else if (size > 64) {
    x = 64;
  } else if (size > 32) {
    x = 32;
  } else if (size > 16) {
    x = 16;
  }
  return [x, 1, 1];
}
