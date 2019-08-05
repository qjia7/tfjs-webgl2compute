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

import {computeDispatch} from '../webgl2compute_util';
import {WebGL2ComputeProgram} from './webgl2compute_program';

export const RELU = 'return max(a, 0.0);';
// TODO(xinghua): Need to verify that x is greater than or equal to 0.
export const LOG = 'return log(a);';

export class UnaryOpProgram implements WebGL2ComputeProgram {
  outputShape: number[];
  userCode: string;
  dispatchLayout: {x: number[]};
  workGroupSize: [number, number, number] = [64, 1, 1];
  dispatch: [number, number, number];
  variableNames = ['A'];

  constructor(op: string, outputShape: number[]) {
    this.outputShape = outputShape;
    this.dispatchLayout = {x: this.outputShape.map((d, i) => i)};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.userCode = `
      float unaryOperation(float a) {
        ${op}
      }

      void main() {
        int index = int(gl_GlobalInvocationID.x);
        float a = getAAtOutCoords();
        setOutput(index, unaryOperation(a));
      }
    `;
  }
}
