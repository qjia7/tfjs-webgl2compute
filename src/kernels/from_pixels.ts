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

import {computeDispatch, flatDispatchLayout} from '../webgl2compute_util';

import {WebGL2ComputeProgram} from './webgl2compute_program';

export class FromPixelsProgram implements WebGL2ComputeProgram {
  outputShape: number[];
  userCode: string;
  variableNames = ['A'];
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number] = [64, 1, 1];

  constructor(outputShape: number[]) {
    const [height, width, ] = outputShape;
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.userCode = `
      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${width}.0, ${height}.0);
        vec4 values = texelFetch(A, uv);
        float value;
        if (depth == 0) {
          value = values.r;
        } else if (depth == 1) {
          value = values.g;
        } else if (depth == 2) {
          value = values.b;
        } else if (depth == 3) {
          value = values.a;
        }
        setOutput(floor(value * 255.0 + 0.5));
      }
    `;
  }
}