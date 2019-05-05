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
import {WebGL2ComputeProgram} from './webgl2compute_math';

export class MultiplyProgram implements WebGL2ComputeProgram {
  outputShape: number[];
  userCode: string;
  dispatch: number[];

  private getWorkGroupXSize() {
    const size = util.sizeFromShape(this.outputShape);
    if (size > 512) return 512;
    if (size > 256) return 256;
    if (size > 128) return 128;
    if (size > 64) return 64;
    if (size > 32) return 32;
    if (size > 16) return 16;
    return 16;
  }

  constructor(outputShape: number[]) {
    this.outputShape = outputShape;
    const workGroupSize = [this.getWorkGroupXSize(), 1, 1];
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
       void main() {
        uint index = gl_GlobalInvocationID.x;
        result[index] = A[index] * B[index];
      }
    `;
  }
}
