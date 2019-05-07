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

import {Tensor, util} from '@tensorflow/tfjs-core';

import * as shader_preprocessor from '../shader_preprocessor';

export interface WebGL2ComputeProgram {
  userCode: string;
  outputShape: number[];
  workGroupSize: [number, number, number];
  dispatch: [number, number, number];
  variableNames: string[];
  uniforms?: string;
}

const lineNumberRegex = /ERROR: [0-9]+:([0-9]+):/g;
function logShaderSourceAndInfoLog(
    shaderSource: string, shaderInfoLog: string) {
  const lineNumberRegexResult = lineNumberRegex.exec(shaderInfoLog);
  if (lineNumberRegexResult == null) {
    console.log(`Couldn't parse line number in error: ${shaderInfoLog}`);
    console.log(shaderSource);
    return;
  }

  const lineNumber = +lineNumberRegexResult[1];

  const shaderLines = shaderSource.split('\n');
  const pad = shaderLines.length.toString().length + 2;
  const linesWithLineNumbers = shaderLines.map(
      (line, lineNumber) =>
          util.rightPad((lineNumber + 1).toString(), pad) + line);
  let maxLineLength = 0;
  for (let i = 0; i < linesWithLineNumbers.length; i++) {
    maxLineLength = Math.max(linesWithLineNumbers[i].length, maxLineLength);
  }

  const beforeErrorLines = linesWithLineNumbers.slice(0, lineNumber - 1);
  const errorLine = linesWithLineNumbers.slice(lineNumber - 1, lineNumber);
  const afterErrorLines = linesWithLineNumbers.slice(lineNumber);

  console.log(beforeErrorLines.join('\n'));
  console.log(shaderInfoLog.split('\n')[0]);
  console.log(
      `%c ${util.rightPad(errorLine[0], maxLineLength)}`,
      'border:1px solid red; background-color:#e3d2d2; color:#a61717');
  console.log(afterErrorLines.join('\n'));
}

export function compileProgram(
    program: WebGL2ComputeProgram, inputs: Tensor[], output: Tensor,
    gl: WebGLRenderingContext): WebGLProgram {
  const inputsData = inputs.map((input: Tensor) => {
    return {dtype: input.dtype, shape: input.shape};
  });
  // const outputData = {dtype: output.dtype, shape: output.shape};
  const source = shader_preprocessor.makeShader(inputsData, program);

  const shader = gl.createShader((gl as any).COMPUTE_SHADER);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (gl.getShaderParameter(shader, gl.COMPILE_STATUS) === false) {
    logShaderSourceAndInfoLog(source, gl.getShaderInfoLog(shader));
    throw new Error('Failed to compile compute shader.');
  }
  const cProg = gl.createProgram();
  gl.attachShader(cProg, shader);
  gl.linkProgram(cProg);
  gl.validateProgram(cProg);
  if (gl.getProgramParameter(cProg, gl.VALIDATE_STATUS) === false) {
    console.log(gl.getProgramInfoLog(cProg));
    throw new Error('Shader program validation failed.');
  }

  return cProg;
}

export function makeShaderKey(program: WebGL2ComputeProgram): string {
  const key = program.workGroupSize.join(',') + program.userCode;
  return key;
}
