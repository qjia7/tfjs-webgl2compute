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
// 4. Built-in function 'dot' is only for genType as arguments. So in
// SAMPLING_SNIPPETS, we can't use dot.
// 5. Unify to use int in case we need to cast int/uint here and there.
// 6. Can't directly return a buffer variable value to workaround an ANGLE bug.

import {DataType} from '@tensorflow/tfjs-core';
import {getBroadcastDims} from '@tensorflow/tfjs-core/dist/ops/broadcast_util';

import {symbolicallyComputeStrides} from './shader_util';

export function getCoordsDataType(rank: number): string {
  if (rank <= 1) {
    return 'int';
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

type GLSLDataType = 'float'|'int'|'vec4';
function mapToGlslTypes(type: DataType, isPacked: boolean): GLSLDataType|
    DataType {
  if (type === 'float32') {
    if (isPacked) {
      return 'vec4';
    }
    return 'float';
  }
  if (type === 'int32') {
    return 'int';
  }
  return type;
}

interface ProgramParams {
  dispatchLayout: {x: number[], y?: number[], z?: number[]};
  workGroupSize: [number, number, number];
  variableNames: string[];
  uniforms?: string;
  userCode: string;
  isPacked?: boolean;
}

interface InputInfo {
  dtype: DataType;
  shape: number[];
  name: string;
}

export function makeShader(
    inputInfo: InputInfo[], outputData: {dtype: DataType, shape: number[]},
    program: ProgramParams): string {
  const prefixSnippets: string[] = [];

  prefixSnippets.push(`
    layout (local_size_x = ${program.workGroupSize[0]},
            local_size_y = ${program.workGroupSize[1]},
            local_size_z = ${program.workGroupSize[2]}) in;
  `);

  let uniformDeclaration = '';
  program.variableNames.forEach((x, i) => {
    uniformDeclaration += `${getCoordsDataType(inputInfo[i].shape.length)} ${
        x.toLowerCase()}Shape; `;
    prefixSnippets.push(`
      layout(std430, binding = ${i}) readonly buffer ssb${x} {
        ${mapToGlslTypes(inputInfo[i].dtype, program.isPacked)} ${x}[];
      };
    `);
  });

  // Output buffer.
  if (program.isPacked) {
    prefixSnippets.push(`
    layout(std430, binding = ${
        program.variableNames.length}) writeonly buffer ssbOut {
      vec4 result[];
    };
  `);
  } else {
    prefixSnippets.push(`
    layout(std430, binding = ${
        program.variableNames.length}) writeonly buffer ssbOut {
        ${mapToGlslTypes(outputData.dtype, false)} result[];
    };
  `);
  }

  uniformDeclaration +=
      `${getCoordsDataType(outputData.shape.length)} outShape; `;

  if (program.uniforms) {
    uniformDeclaration += program.uniforms;
  }

  prefixSnippets.push(`
    layout(std140, binding = 0) uniform Uniforms {
      ${uniformDeclaration}
    };
  `);
  const [getOutputCoords, dispatchLayoutRank] =
      generateGetOutputCoords(program.dispatchLayout);
  if (program.isPacked) {
    const inputSamplingSnippet =
        inputInfo.map(x => getInputSamplingSnippet(x, outputData.shape))
            .join('\n');
    const source = [
      SHADER_PREFIX, prefixSnippets.join('\n'), SAMPLING_SNIPPETS,
      getOutputCoords, inputSamplingSnippet, program.userCode
    ].join('\n');
    return source;
  }
  const sources = [
    SHADER_PREFIX, prefixSnippets.join('\n'), SAMPLING_SNIPPETS,
    getOutputCoords,
    getSetOutputSnippet(outputData.shape.length, outputData.dtype)
  ];
  // TODO: This changes gurantees below test case run correctly. But leads to
  // significant performance regression on some case.
  // tslint:disable-next-line:max-line-length
  // Test case:
  // https://github.com/tensorflow/tfjs-core/pull/1745/files#diff-15289bfca30229483003cbaea418596aR191.
  // Merged from: https://github.com/tensorflow/tfjs-core/pull/1745.
  if (dispatchLayoutRank === outputData.shape.length) {
    // Input sampling snippet is only meaningful when the output isn't getting
    // implicitly reshaped (like it does in conv2d_matmul).
    const inputSamplingSnippet =
        inputInfo.map(x => getInputSamplingSnippet(x, outputData.shape))
            .join('\n');
    sources.push(inputSamplingSnippet);
  }

  sources.push(program.userCode);
  const source = sources.join('\n');
  return source;
}

const SHADER_PREFIX = `#version 310 es
`;

const SAMPLING_SNIPPETS = `
  int getFlatIndex(int coord, int shape) {
    return coord;
  }

  int getFlatIndex(ivec2 coords, ivec2 shape) {
    return coords.x * shape.y + coords.y;
  }

  int getFlatIndex(ivec3 coords, ivec3 shape) {
    int index = coords.x * shape.y * shape.z + coords.y * shape.z + coords.z;
    return index;
  }

  int getFlatIndex(ivec4 coords, ivec4 shape) {
    int index = coords.x * shape.y * shape.z * shape.w +
                coords.y * shape.z * shape.w +
                coords.z * shape.w +
                coords.w;
    return index;
  }
`;

function getSetOutputSnippet(outRank: number, outBufferType: DataType): string {
  let snippet = `void setOutput(int flatIndex, float value) {
      result[flatIndex] = ${
      mapToGlslTypes(outBufferType, false) === 'int' ? 'int(value)' : 'value'};
    }
    void setOutput(int flatIndex, int value) {
      result[flatIndex] = ${
      mapToGlslTypes(outBufferType, false) === 'float' ? 'float(value)' :
                                                         'value'};
    }`;

  if (outRank >= 2) {
    const dims = ['d0', 'd1', 'd2', 'd3'].slice(0, outRank);
    const type = getCoordsDataType(outRank);

    snippet += `
      void setOutput(${dims.map(d => `int ${d}`).join(', ')}, float value) {
        int flatIndex = getFlatIndex(${type}(${dims.join(', ')}), outShape);
        setOutput(flatIndex, value);
      }
      void setOutput(${dims.map(d => `int ${d}`).join(', ')}, int value) {
        int flatIndex = getFlatIndex(${type}(${dims.join(', ')}), outShape);
        setOutput(flatIndex, value);
      }
    `;
  }

  return snippet;
}

function getInputSamplingSnippet(
    inInfo: InputInfo, outShape: number[]): string {
  let res = getSamplerFromInInfo(inInfo);

  const inShape = inInfo.shape;
  if (inShape.length <= outShape.length) {
    res += getSamplerAtOutputCoords(inInfo, outShape);
  }

  return res;
}

function getSamplerFromInInfo(inInfo: InputInfo): string {
  const texName = inInfo.name;
  const rank = inInfo.shape.length;
  const type = getCoordsDataType(rank);
  const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
  const dims = ['d0', 'd1', 'd2', 'd3'].slice(0, rank);
  const inputs = dims.map(d => `int ${d}`).join(', ');

  if (rank < 1) {
    return `
      float ${funcName}() {
        return float(${texName}[0]);
      }
    `;
  }

  return `
    float ${funcName}(${inputs}) {
      return float(${texName}[getFlatIndex(${type}(${dims.join(',')}),
        ${texName.charAt(0).toLowerCase() + texName.slice(1)}Shape)]);
    }
  `;
}
function getSamplerAtOutputCoords(inInfo: InputInfo, outShape: number[]) {
  const texName = inInfo.name;
  const texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);
  const funcName = 'get' + texFuncSnippet + 'AtOutCoords';

  const inRank = inInfo.shape.length;
  const outRank = outShape.length;
  const type = getCoordsDataType(outRank);

  const broadcastDims = getBroadcastDims(inInfo.shape, outShape);
  const rankDiff = outRank - inRank;

  let coordsSnippet = '';

  if (inRank > 0) {
    if (outRank < 2 && broadcastDims.length >= 1) {
      coordsSnippet = 'coords = 0.;';
    } else {
      coordsSnippet =
          broadcastDims.map(d => `coords[${d + rankDiff}] = 0;`).join('\n');
    }
  }

  let unpackedCoordsSnippet = '';
  if (outRank < 2 && inRank > 0) {
    unpackedCoordsSnippet = 'coords';
  } else {
    if (outRank > 1) {
      const coordsType = getCoordsDataType(inRank);
      const coordsValues =
          inInfo.shape.map((s, i) => `coords[${i + rankDiff}]`).join(', ');
      unpackedCoordsSnippet = `${coordsType}(${coordsValues})`;
    } else {
      unpackedCoordsSnippet = 'coords';
    }
  }

  return `
  float ${funcName}() {
    ${type} coords = getOutputCoords();
    ${coordsSnippet}
    float result = float(${texName}[getFlatIndex(${unpackedCoordsSnippet}, ${
      texName.charAt(0).toLowerCase() + texName.slice(1)}Shape)]);
    return result;
  }
`;
}

/**
 * Generates getOutputCoords() function that computes output coordinates from
 * dispatch geometry to reduce arithmetic.
 */

function generateGetOutputCoords(
    dispatchLayout: {x: number[], y?: number[], z?: number[]}):
    [string, number] {
  const {x, y = [], z = []} = dispatchLayout;
  let gatherDimensionsStr = '';
  const dims = [x, y, z];
  let rank = 0;
  for (let i = 0; i < dims.length; i++) {
    const arr = dims[i];

    if (arr.length === 0) {
      continue;
    }
    rank += arr.length;
    if (arr.length === 1) {
      gatherDimensionsStr +=
          `int d${arr[0]} = int(gl_GlobalInvocationID[${i}]);`;
    } else {
      const strides = symbolicallyComputeStrides(arr, 'outShape');
      gatherDimensionsStr += `int index${i} =
      int(gl_GlobalInvocationID[${i}]);`;
      for (let j = 0; j < strides.length; j++) {
        gatherDimensionsStr += `int d${arr[j]} = index${i} / ${strides[j]};`;

        if (j === strides.length - 1) {
          gatherDimensionsStr += `int d${arr[j + 1]} = ` +
              `index${i} - d${arr[j]} * ${strides[j]};`;
        } else {
          gatherDimensionsStr += `index${i} -= d${arr[j]} * ${strides[j]};`;
        }
      }
    }
  }

  const dimensions = [];
  for (let i = 0; i < rank; i++) {
    dimensions.push(`d${i}`);
  }

  const dtype = getCoordsDataType(rank);

  const snippet = `${dtype} getOutputCoords() {
    ${gatherDimensionsStr}
    return ${dtype}(${dimensions.join(',')});
  }`;
  return [snippet, rank];
}
