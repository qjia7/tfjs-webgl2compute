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

import {util} from '@tensorflow/tfjs-core';

import {computeWorkGroupSize} from '../webgl2compute_util';
import {WebGL2ComputeProgram} from './webgl2compute_program';

export const MUL = 'return a * b;';
export const ADD = 'return a + b;';

export class BinaryOpProgram implements WebGL2ComputeProgram {
  outputShape: number[];
  userCode: string;
  dispatch: [number, number, number];

  constructor(op: string, outputShape: number[]) {
    this.outputShape = outputShape;
    const workGroupSize = computeWorkGroupSize(outputShape);
    this.dispatch =
        [Math.ceil(util.sizeFromShape(outputShape) / workGroupSize[0]), 1, 1];

    this.userCode = `#version 310 es
     layout(local_size_x=${workGroupSize[0]}, local_size_y=${
        workGroupSize[1]}) in;
      layout(std430, binding = 0) buffer ssbA {
        float A[];
      };
      layout(std430, binding = 1) buffer ssbB {
        float B[];
      };
      layout(std430, binding = 2) buffer ssbOut {
        float result[];
      };

      float binaryOperation(float a, float b) {
        ${op}
      }

      void main() {
        uint index = gl_GlobalInvocationID.x;
        float a = A[index];
        float b = B[index];
        result[index] = binaryOperation(a, b);
      }
    `;
  }
}
