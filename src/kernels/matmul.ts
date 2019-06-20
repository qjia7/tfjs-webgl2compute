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
// 1. use 'float result = A[*]; return result;' instead of 'return A[*];' to
// workaround an ANGLE bug.
// 2. global variable initializers must be constant expressions. So we can't
// assign the uniform to a global variable.

import {computeDispatch} from '../webgl2compute_util';

import {WebGL2ComputeProgram} from './webgl2compute_program';

export const matMulHeader = `
  float mm_readA(int row, int col);
  float mm_readB(int row, int col);
  void mm_write(int row, int col, float value);
  void mm_matMul(int dimAOuter, int dimInner, int dimBOuter);`;

export function makeMatMulSource(): string {
  return `
    ${matMulHeader}

    const int MatTileSize = int(gl_WorkGroupSize.x);  // .x == .y
    shared float mm_Asub[MatTileSize][MatTileSize];
    shared float mm_Bsub[MatTileSize][MatTileSize];

    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
        int localRow = int(gl_LocalInvocationID.y);  // 0..MatTileSize
        int localCol = int(gl_LocalInvocationID.x);  // 0..MatTileSize
        int globalRow = int(gl_GlobalInvocationID.y);  // AOuter
        int globalCol = int(gl_GlobalInvocationID.x);  // Inner

        float acc = 0.0;

        int numTiles = (dimInner - 1) / MatTileSize + 1;

        for (int t = 0; t < numTiles; t++) {
          // Load one tile of A and B into local memory
          int tiledACol = MatTileSize * t + localCol;
          int tiledBRow = MatTileSize * t + localRow;
          mm_Asub[localRow][localCol] = mm_readA(globalRow, tiledACol);
          mm_Bsub[localRow][localCol] = mm_readB(tiledBRow, globalCol);

          // Synchronise to make sure the tile is loaded
          barrier();

          for (int k = 0; k < MatTileSize; k++) {
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

      float mm_readA(int row, int col) {
        if (row < aShape[1] && col < aShape[2]) {
          float result = A[row * aShape[2] + col];
          return result;
        } else {
          return 0.0;
        }
      }

      float mm_readB(int row, int col) {
        if (row < aShape[2] && col < bShape[2]) {
          float result = B[row * bShape[2] + col];
          return result;
        } else {
          return 0.0;
        }
      }

      void mm_write(int row, int col, float value) {
        setOutput(row * bShape[2] + col, value);
      }

      void main() {
        mm_matMul(aShape[1], aShape[2], bShape[2]);
      }
    `;
  }
}
