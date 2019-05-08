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

import {WebGL2ComputeProgram} from './webgl2compute_program';

export class MatMulProgram implements WebGL2ComputeProgram {
  outputShape: number[];
  userCode: string;
  workGroupSize: [number, number, number] = [16, 16, 1];
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  // M is A outer, N is shared, K is B outer
  uniforms = 'uint M, N, K, batch;';

  constructor(outputShape: [number, number, number]) {
    this.outputShape = outputShape;
    this.dispatch = [
      Math.ceil(outputShape[1] / this.workGroupSize[0]),
      Math.ceil(outputShape[2] / this.workGroupSize[1]), 1
    ];

    this.userCode = `
      const uint TileSize = gl_WorkGroupSize.x;
      shared float Asub[TileSize][TileSize];
      shared float Bsub[TileSize][TileSize];

      void main() {
        uint row = gl_LocalInvocationID.y; // Local row ID (max: TileSize)
        uint col = gl_LocalInvocationID.x; // Local col ID (max: TileSize)
        uint globalRow = TileSize*gl_WorkGroupID.y + row; // Row ID of C (0..M)
        uint globalCol = TileSize*gl_WorkGroupID.x + col; // Col ID of C (0..N)

        float acc = 0.0;

        // Add 1 to N to ceil.
        uint numTiles = (N % TileSize) != 0u ? N/TileSize + 1u : N/TileSize;

        for (uint t=0u; t<numTiles; t++) {
          // Load one tile of A and B into local memory
          uint tiledRow = TileSize*t + row;
          uint tiledCol = TileSize*t + col;
          Asub[row][col] = A[globalRow*N + tiledCol];
          Bsub[row][col] = B[tiledRow*K + globalCol];

          // Synchronise to make sure the tile is loaded
          // memoryBarrierShared();
          barrier();
          uint sizeTS =
              (t == (numTiles - 1u) && ((N % TileSize) != 0u)) ? N % TileSize :  TileSize;
          for (uint k=0u; k<sizeTS; k++) {
            acc += Asub[row][k] * Bsub[k][col];
          }

          // Synchronise before loading the next tile
          barrier();
        }

        if(globalCol < K && globalRow < M) {
          setOutput(globalRow*K + globalCol, acc);
        }
      }
    `;
  }
}
