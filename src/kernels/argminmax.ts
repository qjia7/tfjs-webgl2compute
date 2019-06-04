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
// Remove variables' const qualifier;
// Explicitly convert int to uint;
// Explicitly convert result from any type to float type.

import {util} from '@tensorflow/tfjs-core';
import * as axis_util from '@tensorflow/tfjs-core/dist/ops/axis_util';

import {getCoordsDataType} from '../shader_preprocessor';
import {computeDispatch} from '../webgl2compute_util';

import {WebGL2ComputeProgram} from './webgl2compute_program';

export class ArgMinMaxProgram implements WebGL2ComputeProgram {
  outputShape: number[];
  userCode: string;
  dispatchLayout: {x: number[], y: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number];
  variableNames = ['x'];
  uniforms = 'uint axis;';

  constructor(inputShape: number[], axis: number, reduceType: 'min'|'max') {
    const axes = [axis];
    axis_util.assertAxesAreInnerMostDims(
        'arg' + reduceType.charAt(0).toUpperCase() + reduceType.slice(1), axes,
        inputShape.length);

    const op = reduceType === 'min' ? '<' : '>';

    // |outShape| is the shape with the removed axis
    // |reduceShape| is the shape we are reducing. i.e. [ inputShape[axis] ]
    const [outputShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(inputShape, axes);

    this.outputShape = outputShape.length === 0 ? [1] : outputShape;

    // Length of the axis we're reducing on.
    const reduceSize = util.sizeFromShape(reduceShape);

    // The number of comparisons each thread will do
    const reductionFactor = 2;
    const xMaxThreads = 1024;  // gl_MaxComputeWorkGroupSize
    const xThreads =
        Math.min(Math.ceil(reduceSize / reductionFactor), xMaxThreads);

    this.workGroupSize = [xThreads, 1, 1];

    this.dispatchLayout = {x: [], y: this.outputShape.map((d, i) => i)};
    this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape);

    // When xThreads > 1, each thread reduces Length / xThreads values.
    // Thes results are stored in shared memory and iteratively reduced.
    const reduceInSharedMemory = xThreads > 1;
    const sharedMemorySnippet = `
      shared uint xBestIndices[WorkGroupSize];
      shared float xBestValues[WorkGroupSize];
    `;

    const sharedMemoryReduceSnippet = `
      xBestIndices[gl_LocalInvocationID.x] = bestIndex;
      xBestValues[gl_LocalInvocationID.x] = bestValue;

      uint currentSize = WorkGroupSize;
      while (currentSize > 1u) {
        barrier();

        for (uint w = 0u; w < ${reductionFactor}u; ++w) {
          uint i = gl_LocalInvocationID.x * ${reductionFactor}u + w;
          if (i < currentSize) {
            uint candidateIndex = xBestIndices[i];
            float candidate = xBestValues[i];
            if (candidate ${op} bestValue && !isnan(candidate)) {
              bestValue = candidate;
              bestIndex = candidateIndex;
            }
          }
        }

        xBestIndices[gl_LocalInvocationID.x] = bestIndex;
        xBestValues[gl_LocalInvocationID.x] = bestValue;

        currentSize = DIV_CEIL(currentSize, ${reductionFactor}u);
      }

      if (gl_LocalInvocationID.x == 0u) {
        setOutput(int(flatOutputIndex), float(bestIndex));
      }
    `;

    const outputCoordsType = getCoordsDataType(this.outputShape.length);

    const indexOutputCoords = (outputCoords: string, index: string) => {
      if (this.outputShape.length === 1) {
        return outputCoords;
      } else {
        return `${outputCoords}[${index}]`;
      }
    };

    const indexInputShape = (index: string) => {
      if (inputShape.length === 1) {
        return 'xShape';
      } else {
        return `xShape[${index}]`;
      }
    };

    this.userCode = `
      #define DIV_CEIL(x, y) (((x) - 1u) / (y) + 1u)

      const uint WorkGroupSize = gl_WorkGroupSize.x;

      ${reduceInSharedMemory ? sharedMemorySnippet : ''}

      // In order to get a flattened index into the input tensor, we need to
      // add back the index along the reduced dimension to |outputCoords|.
      // This function outputs the offset to the first value along
      // |axis| and the stride to get the next value of the input along |axis|.
      uvec2 getInputCoordInfo() {
        ${outputCoordsType} outputCoords = getOutputCoords();
        uint i = ${this.outputShape.length - 1}u;

        uint stride = 1u;
        uint inputStride = 1u;
        uint offset = 0u;

        for (uint r = 1u; r <= ${inputShape.length}u; ++r) {
          uint length = uint(${indexInputShape(`${inputShape.length}u - r`)});
          if (${inputShape.length}u - r == axis) {
            inputStride = stride;
          } else {
            offset += uint(
              ${indexOutputCoords('outputCoords', 'i--')}) * stride;
          }
          stride *= length;
        }

        return uvec2(offset, inputStride);
      }

      uint getInputIndex(uvec2 coordInfo, uint index) {
        return coordInfo[0] + coordInfo[1] * index;
      }

      void main() {
        uvec2 coordInfo = getInputCoordInfo();

        uint bestIndex = 0u;
        float bestValue = float(x[getInputIndex(coordInfo, bestIndex)]);

        uint Length = uint(${indexInputShape('axis')});
        uint WorkPerThread = DIV_CEIL(Length, WorkGroupSize);

        for (uint w = 0u; w < WorkPerThread; ++w) {
          uint i = gl_GlobalInvocationID.x * WorkPerThread + w;
          if (i < Length) {
            float candidate = float(x[getInputIndex(coordInfo, i)]);
            if (candidate ${op} bestValue && !isnan(candidate)) {
              bestValue = candidate;
              bestIndex = i;
            }
          }
        }

        uint flatOutputIndex = gl_GlobalInvocationID.y;
        ${
        reduceInSharedMemory ?
            sharedMemoryReduceSnippet :
            'setOutput(int(flatOutputIndex), float(bestIndex));'}
      }
    `;
  }
}
