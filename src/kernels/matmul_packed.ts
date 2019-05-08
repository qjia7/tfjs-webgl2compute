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

import {computeDispatch} from '../webgl2compute_util';
import {matMulHeader} from './matmul';
import {WebGL2ComputeProgram} from './webgl2compute_program';

export function makeMatMulPackedSource(workPerThread: number): string {
  return `
    ${matMulHeader}

    const uint WorkGroupSize = gl_WorkGroupSize.x;  // .x == .y
    const uint WorkPerThread = ${workPerThread}u;
    const uint MatTileSize = WorkGroupSize * WorkPerThread;

    shared float mm_Asub[MatTileSize][MatTileSize];
    shared float mm_Bsub[MatTileSize][MatTileSize];

    void mm_matMul(uint dimAOuter, uint dimInner, uint dimBOuter) {
      // These are 0..MatTileSize, in increments of WorkPerThread.
      uint tileRow = gl_LocalInvocationID.y * WorkPerThread;
      uint tileCol = gl_LocalInvocationID.x * WorkPerThread;

      // These are 0..AOuter, in increments of WorkPerThread.
      uint globalRow = gl_GlobalInvocationID.y * WorkPerThread;
      uint globalCol = gl_GlobalInvocationID.x * WorkPerThread;

      uint numTiles = (dimInner - 1u) / MatTileSize + 1u;

      float acc[WorkPerThread][WorkPerThread];
      float ACached;
      float BCached[WorkPerThread];

      // Without this initialization strange values show up in acc.
      for (uint innerRow = 0u; innerRow < WorkPerThread; innerRow++) {
        for (uint innerCol = 0u; innerCol < WorkPerThread; innerCol++) {
          acc[innerRow][innerCol] = 0.0;
        }
      }

      // Loop over shared dimension.
      for (uint t = 0u; t < numTiles; t++) {
        // Load one tile of A and B into local memory.
        for (uint innerRow = 0u; innerRow < WorkPerThread; innerRow++) {
          for (uint innerCol = 0u; innerCol < WorkPerThread; innerCol++) {
            uint inputRow = tileRow + innerRow;
            uint inputCol = tileCol + innerCol;

            mm_Asub[inputRow][inputCol] = mm_readA(
                globalRow + innerRow,
                t * MatTileSize + tileCol + innerCol);
            mm_Bsub[inputRow][inputCol] = mm_readB(
                t * MatTileSize + tileRow + innerRow,
                globalCol + innerCol);
          }
        }

        barrier();

        // Compute acc values for a single thread.
        for (uint k = 0u; k < MatTileSize; k++) {
          for (uint inner = 0u; inner < WorkPerThread; inner++) {
            BCached[inner] = mm_Bsub[k][tileCol + inner];
          }

          for (uint innerRow = 0u; innerRow < WorkPerThread; innerRow++) {
            ACached = mm_Asub[tileRow + innerRow][k];
            for (uint innerCol = 0u; innerCol < WorkPerThread; innerCol++) {
              acc[innerRow][innerCol] += ACached * BCached[innerCol];
            }
          }
        }

        barrier();
      }

      for (uint innerRow = 0u; innerRow < WorkPerThread; innerRow++) {
        for (uint innerCol = 0u; innerCol < WorkPerThread; innerCol++) {
          uint globalFlatIndex =
            (globalRow + innerRow) * dimBOuter + (globalCol + innerCol);

          if ((globalCol + innerCol) < dimBOuter &&
              (globalRow + innerRow) < dimAOuter) {
            mm_write(globalRow + innerRow,
                     globalCol + innerCol,
                     acc[innerRow][innerCol]);
          }
        }
      }
    }
  `;
}

export class MatMulPackedProgram implements WebGL2ComputeProgram {
  outputShape: number[];
  userCode: string;
  dispatch: [number, number, number];
  workPerThread: number;
  variableNames = ['A', 'B'];
  uniforms = 'uint dimAOuter, dimInner, dimBOuter, batch;';
  workGroupSize: [number, number, number] = [16, 16, 1];

  constructor(outputShape: [number, number, number], workPerThread: number) {
    this.outputShape = outputShape;
    this.workPerThread = workPerThread;

    const dispatchLayout = {x: [1], y: [2], z: [0]};
    this.dispatch = computeDispatch(
        dispatchLayout, this.outputShape, this.workGroupSize,
        [workPerThread, workPerThread, 1]);

    // Consider compiling a different version of the shader that doesn't care
    // about boundary conditions when loading from Asub / Bsub when tiles fit
    // neatly inside of output. May slightly improve performance.
    this.userCode = `
      ${makeMatMulPackedSource(workPerThread)}

      float mm_readA(uint row, uint col) {
        if (row < dimAOuter && col < dimInner) {
          float result = A[row * dimInner + col];
          return result;
        } else {
          return 0.0;
        }
      }

      float mm_readB(uint row, uint col) {
        if (row < dimInner && col < dimBOuter) {
          float result = B[row * dimBOuter + col];
          return result;
        } else {
          return 0.0;
        }
      }

      void mm_write(uint row, uint col, float value) {
        setOutput(row * dimBOuter + col, value);
      }

      void main() {
        mm_matMul(dimAOuter, dimInner, dimBOuter);
      }
    `;
  }
}
