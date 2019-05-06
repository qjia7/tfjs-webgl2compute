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

import {DataType} from '@tensorflow/tfjs-core';

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

export function makeShader(
    inputTypes: Array<{dtype: DataType, shape: number[]}>,
    variableNames: string[], userCode: string,
    workGroupSize: [number, number, number], tileSize: number): string {
  let tileSizeSnippet: string;
  if (tileSize != null) {
    tileSizeSnippet = `
    #define TileSize ${tileSize}u
    layout (local_size_x = TileSize, local_size_y = TileSize,
      local_size_z = 1) in;`;
  } else {
    tileSizeSnippet = `
    layout (local_size_x = ${workGroupSize[0]}, local_size_y = ${
        workGroupSize[1]},
      local_size_z = ${workGroupSize[2]}) in;`;
  }
  const prefixSnippets: string[] = [];
  variableNames.forEach((x, i) => {
    prefixSnippets.push(`
      layout(std430, binding = ${i}) readonly buffer ssb${x} {
        ${mapToGlslTypes(inputTypes[i].dtype)} ${x}[];
      };
    `);
  });

  // Output buffer.
  prefixSnippets.push(`
    layout(std430, binding = ${variableNames.length}) writeonly buffer ssbOut {
      float result[];
    };
  `);

  const source = [
    SHADER_PREFIX, tileSizeSnippet, prefixSnippets.join('\n'),
    SET_OUTPUT_SNIPPET, userCode
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
