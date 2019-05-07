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
// 1. SHADER_PREFIX is differ. For wegl2compute, we use '#version 310 es'
// 2. Work group layout 'layout (local_size_x, ...)' is a MUST not optional.
// 3. The qualifier 'set' in buffer layout is not supported in ESSL 310.
// 3. Built-in function 'dot' is only for genType as arguments. So in
// SAMPLING_SNIPPETS, we can't use dot.

import {DataType} from '@tensorflow/tfjs-core';

export function getCoordsDataType(rank: number): string {
  if (rank <= 1) {
    return 'uint';
  } else if (rank === 2) {
    return 'ivec2';
  } else if (rank === 3) {
    return 'ivec3';
  } else if (rank === 4) {
    return 'ivec4';
  } else {
    throw Error(`GPU for rank ${rank} is not yet supported`);
  }
}

type GLSLDataType = 'float'|'uint';
function mapToGlslTypes(type: DataType): GLSLDataType|DataType {
  if (type === 'float32') {
    return 'float';
  }
  if (type === 'int32') {
    return 'uint';
  }
  return type;
}

interface ProgramParams {
  workGroupSize: [number, number, number];
  variableNames: string[];
  uniforms?: string;
  userCode: string;
}

export function makeShader(
    inputTypes: Array<{dtype: DataType, shape: number[]}>,
    program: ProgramParams): string {
  const prefixSnippets: string[] = [];
  prefixSnippets.push(`
    layout (local_size_x = ${program.workGroupSize[0]},
            local_size_y = ${program.workGroupSize[1]},
            local_size_z = ${program.workGroupSize[2]}) in;
    `);

  program.variableNames.forEach((x, i) => {
    prefixSnippets.push(`
      layout(std430, binding = ${i}) readonly buffer ssb${x} {
        ${mapToGlslTypes(inputTypes[i].dtype)} ${x}[];
      };
    `);
  });

  // Output buffer.
  prefixSnippets.push(`
    layout(std430, binding = ${
      program.variableNames.length}) writeonly buffer ssbOut {
      float result[];
    };
  `);

  if (program.uniforms) {
    prefixSnippets.push(`
      layout(std140, binding = 0) uniform Uniforms {
        ${program.uniforms}
      };
    `);
  }

  const source = [
    SHADER_PREFIX, prefixSnippets.join('\n'), SAMPLING_SNIPPETS,
    SET_OUTPUT_SNIPPET, program.userCode
  ].join('\n');
  return source;
}

const SET_OUTPUT_SNIPPET = `
  void setOutput(uint flatIndex, float value) {
    result[flatIndex] = value;
  }
`;
const SHADER_PREFIX = `#version 310 es
`;

const SAMPLING_SNIPPETS = `
  uint getFlatIndex(uint coord, uint shape) {
    return coord;
  }

  uint getFlatIndex(ivec2 coords, ivec2 shape) {
    return uint(coords.x * shape.y + coords.y);
  }

  uint getFlatIndex(ivec3 coords, ivec3 shape) {
    int index = coords.x * shape.y * shape.z + coords.y * shape.z + coords.z;
    return uint(index);
  }

  uint getFlatIndex(ivec4 coords, ivec4 shape) {
    int index = coords.x * shape.y * shape.z * shape.w +
                coords.y * shape.z * shape.w +
                coords.z * shape.w +
                coords.w;
    return uint(index);
  }
`;
