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
// 1. Add postfix 'u' for uint const.
// 2. use 'float result = A[*]; return result;' instead of 'return A[*];' to
// workaround an ANGLE bug.
// 3. global variable initializers must be constant expressions. So we can't
// assign the uniform to a global variable.

import {computeDispatch} from '../webgl2compute_util';

import {WebGL2ComputeProgram} from './webgl2compute_program';

export const matMulHeader = `
  float mm_readA(uint row, uint col);
  float mm_readB(uint row, uint col);
  void mm_write(uint row, uint col, float value);
  void mm_matMul(uint dimAOuter, uint dimInner, uint dimBOuter);`;

export function makeMatMulSource(): string {
  return `
    ${matMulHeader}

    const uint MatTileSize = gl_WorkGroupSize.x;  // .x == .y
    shared float mm_Asub[MatTileSize][MatTileSize];
    shared float mm_Bsub[MatTileSize][MatTileSize];

    void mm_matMul(uint dimAOuter, uint dimInner, uint dimBOuter) {
        uint localRow = gl_LocalInvocationID.y;  // 0..MatTileSize
        uint localCol = gl_LocalInvocationID.x;  // 0..MatTileSize
        uint globalRow = gl_GlobalInvocationID.y;  // AOuter
        uint globalCol = gl_GlobalInvocationID.x;  // Inner

        float acc = 0.0;

        uint numTiles = (dimInner - 1u) / MatTileSize + 1u;

        for (uint t = 0u; t < numTiles; t++) {
          // Load one tile of A and B into local memory
          uint tiledACol = MatTileSize * t + localCol;
          uint tiledBRow = MatTileSize * t + localRow;
          mm_Asub[localRow][localCol] = mm_readA(globalRow, tiledACol);
          mm_Bsub[localRow][localCol] = mm_readB(tiledBRow, globalCol);

          // Synchronise to make sure the tile is loaded
          barrier();

          for (uint k = 0u; k < MatTileSize; k++) {
            acc += mm_Asub[localRow][k] * mm_Bsub[k][localCol];
          }

          // Synchronise before loading the next tile
          barrier();
        }

        if (globalCol < dimBOuter && globalRow < dimAOuter) {
          mm_write(globalRow, globalCol, acc);
        }
      }
  `;
}

export class MatMulProgram implements WebGL2ComputeProgram {
  outputShape: number[];
  userCode: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  workGroupSize: [number, number, number] = [16, 16, 1];  // Must be square.

  constructor(outputShape: [number, number, number]) {
    this.outputShape = outputShape;
    this.dispatchLayout = {x: [1], y: [2], z: [0]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.userCode = `
      ${makeMatMulSource()}

      float mm_readA(uint row, uint col) {
        int r = int(row);
        int c = int(col);
        if (r < aShape[1] && c < aShape[2]) {
          float result = A[r * aShape[2] + c];
          return result;
        } else {
          return 0.0;
        }
      }

      float mm_readB(uint row, uint col) {
        int r = int(row);
        int c = int(col);
        if (r < aShape[2] && c < bShape[2]) {
          float result = B[r * bShape[2] + c];
          return result;
        } else {
          return 0.0;
        }
      }

      void mm_write(uint row, uint col, float value) {
        setOutput(row * uint(dimBOuter) + col, value);
      }

      void main() {
        mm_matMul(uint(aShape[1]), uint(aShape[2]), uint(bShape[2]));
      }
    `;
  }
}
