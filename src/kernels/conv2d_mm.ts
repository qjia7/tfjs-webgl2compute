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
// Use 0. instead of 0 for float type.

import * as tf from '@tensorflow/tfjs-core';
import {Conv2DInfo} from '@tensorflow/tfjs-core/dist/ops/conv_util';
import {computeDispatch} from '../webgl2compute_util';

import {makeMatMulSource} from './matmul';
import {makeMatMulPackedSource} from './matmul_packed';
import {WebGL2ComputeProgram} from './webgl2compute_program';

export class Conv2DMMProgram implements WebGL2ComputeProgram {
  outputShape: number[];
  userCode: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms = 'ivec2 filterDims, pad, stride;';
  workGroupSize: [number, number, number] = [
    16, 16,  // must be square (for matmul)
    1
  ];

  constructor(convInfo: Conv2DInfo, workPerThread: number) {
    this.outputShape = convInfo.outShape;

    tf.util.assert(
        convInfo.dataFormat === 'channelsLast',
        () => 'TODO: NCHW is unimplemented');
    tf.util.assert(
        convInfo.dilationHeight === 1 && convInfo.dilationWidth === 1,
        () => 'TODO: Dilation is unimplemented');

    let elementsPerThread: [number, number, number];
    let matMulSource: string;
    if (workPerThread === 0) {
      elementsPerThread = [1, 1, 1];
      matMulSource = makeMatMulSource();
    } else {
      elementsPerThread = [workPerThread, workPerThread, 1];
      matMulSource = makeMatMulPackedSource(workPerThread);
    }

    this.dispatchLayout = {x: [1], y: [2], z: [0]};
    const matMulOutShape = [
      convInfo.outShape[0], convInfo.outShape[1] * convInfo.outShape[2],
      convInfo.outShape[3]
    ];

    this.dispatch = computeDispatch(
        this.dispatchLayout, matMulOutShape, this.workGroupSize,
        elementsPerThread);

    this.userCode = `
        ${matMulSource}

        bool coordIsValid(ivec4 coord, ivec4 shape) {
          return all(greaterThanEqual(coord, ivec4(0))) &&
              all(lessThan(coord, shape));
        }

        int batch;

        float mm_readA(int row, int col) {
          ivec4 coord = ivec4(
              (col / filterDims[1]) % filterDims[0],
              col % filterDims[1],
              col / (filterDims[1] * filterDims[0]),
              row);

          ivec4 shape = ivec4(filterDims, xShape[3], outShape[3]);
          return coordIsValid(coord, shape) ?
              W[getFlatIndex(coord, shape)] : 0.;
        }

        float mm_readB(int row, int col) {
          int outRow = col / outShape[2];
          int outCol = col % outShape[2];

          int WRow = (row / filterDims[1]) % filterDims[0];
          int WCol = row % filterDims[1];

          ivec4 coord = ivec4(
              batch,
              pad[0] + outRow * stride[0] + WRow,
              pad[1] + outCol * stride[1] + WCol,
              row / (filterDims[1] * filterDims[0]));
          return coordIsValid(coord, xShape) ?
              x[getFlatIndex(coord, xShape)] : 0.;
        }

        void mm_write(int row, int col, float value) {
          ivec4 outCoord = ivec4(
              batch,
              col / outShape[2],
              col % outShape[2],
              row);
          if (coordIsValid(outCoord, outShape)) {
            result[getFlatIndex(outCoord, outShape)] = value;
          }
        }

        void main() {
          batch = getOutputCoords()[0];

          int dimAOuter = outShape[3];
          int dimBOuter = outShape[1] * outShape[2];
          int dimInner = filterDims[0] * filterDims[1] * xShape[3];
          mm_matMul(dimAOuter, dimInner, dimBOuter);
        }
      `;
  }
}
