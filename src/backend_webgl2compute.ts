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

import {DataMover, DataType, KernelBackend, Rank, ShapeMap, Tensor, Tensor2D, Tensor3D, Tensor4D, util} from '@tensorflow/tfjs-core';
import {computeOutShape} from '@tensorflow/tfjs-core/dist/ops/concat_util';
import {Conv2DInfo} from '@tensorflow/tfjs-core/dist/ops/conv_util';
import {upcastType} from '@tensorflow/tfjs-core/dist/types';

import {ArgMinMaxProgram} from './kernels/argminmax';
import * as binary_op from './kernels/binary_op';
import {BinaryOpProgram} from './kernels/binary_op';
import {ConcatProgram} from './kernels/concat';
import {Conv2DMMProgram} from './kernels/conv2d_mm';
import {Conv2DNaiveProgram} from './kernels/conv2d_naive';
import {MatMulProgram} from './kernels/matmul';
import {MatMulPackedProgram} from './kernels/matmul_packed';
import {MaxPoolProgram} from './kernels/maxpool';
import {TransposeProgram} from './kernels/transpose';
import * as unary_op from './kernels/unary_op';
import {UnaryOpProgram} from './kernels/unary_op';
import * as webgl2compute_math from './kernels/webgl2compute_program';

type TensorInfo = {
  shape: number[],
  dtype: DataType,
  values: Float32Array|Int32Array|Uint8Array,
  id: number,
  buffer?: WebGLBuffer
};

interface DataId {}

const WEBGL_ATTRIBUTES: WebGLContextAttributes = {
  alpha: false,
  antialias: false,
  premultipliedAlpha: false,
  preserveDrawingBuffer: false,
  depth: false,
  stencil: false,
  failIfMajorPerformanceCaveat: true
};

function getWebGLContext(): WebGLRenderingContext {
  const canvas = document.createElement('canvas');
  return canvas.getContext('webgl2-compute', WEBGL_ATTRIBUTES) as
      WebGLRenderingContext;
}

export class WebGL2ComputeBackend extends KernelBackend {
  gl: WebGLRenderingContext;
  private tensorMap = new WeakMap<DataId, TensorInfo>();
  private binaryCache: {[key: string]: WebGLProgram};

  constructor() {
    super();
    this.gl = getWebGLContext();
    this.binaryCache = {};
  }

  floatPrecision(): 16|32 {
    return 32;
  }

  setDataMover(dataMover: DataMover): void {
    // TODO: tfjs team to implement this.
  }

  disposeData(dataId: DataId): void {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }

    const info = this.tensorMap.get(dataId);
    this.destroyBuffer(info.buffer);
  }

  private destroyBuffer(buffer: WebGLBuffer) {
    // TODO: recycle deleted buffers
    this.gl.deleteBuffer(buffer);
  }

  register(dataId: object, shape: number[], dtype: DataType): void {
    if (!this.tensorMap.has(dataId)) {
      const buffer = this.gl.createBuffer();
      // tslint:disable-next-line:no-any
      this.gl.bindBuffer((this.gl as any).SHADER_STORAGE_BUFFER, buffer);
      this.gl.bufferData(
          (this.gl as any).SHADER_STORAGE_BUFFER,
          util.sizeFromShape(shape) * util.bytesPerElement(dtype),
          this.gl.STATIC_DRAW);
      this.tensorMap.set(dataId, {shape, dtype, values: null, id: -1, buffer});
    }
  }

  write(dataId: object, values: Float32Array|Int32Array|Uint8Array): void {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }

    const info = this.tensorMap.get(dataId);
    info.values = values;
    this.gl.bindBuffer((this.gl as any).SHADER_STORAGE_BUFFER, info.buffer);
    this.gl.bufferSubData((this.gl as any).SHADER_STORAGE_BUFFER, 0, values);
  }

  async read(dataId: object): Promise<Float32Array|Int32Array|Uint8Array> {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }
    const info = this.tensorMap.get(dataId);
    const size =
        util.sizeFromShape(info.shape) * util.bytesPerElement(info.dtype);
    // tslint:disable-next-line:no-any
    this.gl.bindBuffer((this.gl as any).SHADER_STORAGE_BUFFER, info.buffer);
    const data = new Float32Array(size / 4);
    (this.gl as any)
        .getBufferSubData((this.gl as any).SHADER_STORAGE_BUFFER, 0, data);

    return data;
  }

  private makeOutputArray<T extends Tensor>(shape: number[], dtype: DataType):
      T {
    return Tensor.make(shape, {}, dtype, this) as T;
  }

  private getAndSaveBinary(key: string, getBinary: () => WebGLProgram):
      WebGLProgram {
    if (!(key in this.binaryCache)) {
      this.binaryCache[key] = getBinary();
    }
    return this.binaryCache[key];
  }

  private compileAndRun<
      K extends {dtype: DataType, size: number, dataId: {}, shape: number[]}>(
      program: webgl2compute_math.WebGL2ComputeProgram, inputs: Tensor[],
      output?: Tensor, programUniforms?: number[]): K {
    if (output == null) {
      output = this.makeOutputArray(program.outputShape, inputs[0].dtype);
    }

    let dimUniforms: number[] = [];
    const bufferShapes = inputs.concat(output).map(d => d.shape);
    bufferShapes.forEach((d, i) => {
      // TODO: handle vec3 uniform upload in a principled way.
      // vec3 and vec4 have the same alignment, however padding is only
      // sometimes necessary. Complete std140 layout rules are documented here:
      // tslint:disable-next-line:max-line-length
      // https://www.khronos.org/registry/OpenGL/specs/gl/glspec45.core.pdf#page=159
      if (d.length === 3 && i > 0 && bufferShapes[i - 1].length === 3) {
        dimUniforms.push(0);
      }
      dimUniforms.push(...d);
    });

    if (programUniforms) {
      dimUniforms = dimUniforms.concat(programUniforms);
    }

    const key = webgl2compute_math.makeShaderKey(
        program, bufferShapes.map(d => d.length));
    const binary = this.getAndSaveBinary(key, () => {
      return webgl2compute_math.compileProgram(
          program, inputs, output, this.gl);
    });
    this.gl.useProgram(binary);

    const uniformData = new Int32Array(dimUniforms);
    // TODO: Create the uniform buffer when the program is created. And update
    // uniform buffer when use the program.
    let uniformBuffer;
    uniformBuffer = this.makeUniforms(uniformData);

    let outputBinding = 0;
    inputs.forEach((input, i) => {
      const mapInfo = this.tensorMap.get(input.dataId);
      (this.gl as any)
          .bindBufferBase(
              (this.gl as any).SHADER_STORAGE_BUFFER, i, mapInfo.buffer);
      outputBinding = outputBinding + 1;
    });

    const mapInfo = this.tensorMap.get(output.dataId);
    (this.gl as any)
        .bindBufferBase(
            (this.gl as any).SHADER_STORAGE_BUFFER, outputBinding,
            mapInfo.buffer);

    (this.gl as any)
        .dispatchCompute(
            program.dispatch[0], program.dispatch[1], program.dispatch[2]);
    if (programUniforms) {
      this.destroyBuffer(uniformBuffer);
    }
    return output as {} as K;
  }

  private makeUniforms(data: Uint32Array|Int32Array): WebGLBuffer {
    const buffer = this.gl.createBuffer();
    // tslint:disable-next-line:no-any
    this.gl.bindBuffer((this.gl as any).UNIFORM_BUFFER, buffer);
    this.gl.bufferData(
        (this.gl as any).UNIFORM_BUFFER, data, this.gl.STATIC_DRAW);
    (this.gl as any).bindBufferBase((this.gl as any).UNIFORM_BUFFER, 0, buffer);
    return buffer;
  }

  private binaryOp(a: Tensor, b: Tensor, op: string) {
    const dtype = upcastType(a.dtype, b.dtype);
    const program = new BinaryOpProgram(op, a.shape, b.shape);
    const output = Tensor.make(program.outputShape, {}, dtype) as Tensor;

    const result = this.compileAndRun(program, [a, b], output) as Tensor;
    return result;
  }

  add(a: Tensor, b: Tensor): Tensor {
    return this.binaryOp(a, b, binary_op.ADD);
  }

  multiply(a: Tensor, b: Tensor): Tensor {
    return this.binaryOp(a, b, binary_op.MUL);
  }

  relu<T extends Tensor>(x: T): T {
    const program = new UnaryOpProgram(unary_op.RELU, x.shape);
    return this.compileAndRun(program, [x]) as T;
  }

  reshape<R extends Rank>(x: Tensor, shape: ShapeMap[R]): Tensor<R> {
    return Tensor.make(shape, {dataId: x.dataId}, x.dtype);
  }

  batchMatMul(
      a: Tensor3D, b: Tensor3D, transposeA: boolean,
      transposeB: boolean): Tensor3D {
    const outerShapeA = transposeA ? a.shape[2] : a.shape[1];
    const outerShapeB = transposeB ? b.shape[1] : b.shape[2];
    const [batch, , ] = a.shape;

    const output =
        Tensor.make([batch, outerShapeA, outerShapeB], {}, a.dtype, this) as
        Tensor3D;

    let program: MatMulProgram|MatMulPackedProgram;
    program = new MatMulPackedProgram(output.shape, 4);

    return this.compileAndRun(program, [a, b], output) as Tensor3D;
  }

  conv2d(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    const output =
        Tensor.make(convInfo.outShape, {}, x.dtype, this) as Tensor4D;
    let program: Conv2DNaiveProgram|Conv2DMMProgram;
    program = new Conv2DMMProgram(convInfo, 4);

    const pad = convInfo.padInfo.type === 'VALID' ?
        [0, 0] :
        convInfo.padInfo.type === 'SAME' ?
        [
          -Math.floor((convInfo.filterShape[0] - 1) / 2),
          -Math.floor((convInfo.filterShape[1] - 1) / 2)
        ] :
        [convInfo.padInfo.top, convInfo.padInfo.left];

    const dimensions = [
      convInfo.filterHeight,
      convInfo.filterWidth,
      ...pad,
      convInfo.strideHeight,
      convInfo.strideWidth,
    ];

    const result = this.compileAndRun(
                       program, [x, filter], output, dimensions) as Tensor4D;
    return result;
  }

  transpose<T extends Tensor>(x: T, perm: number[]): T {
    const program = new TransposeProgram(x.shape, perm);
    return this.compileAndRun(program, [x]);
  }

  private argMinMaxReduce(x: Tensor, axis: number, reduceType: 'min'|'max'):
      Tensor {
    const program = new ArgMinMaxProgram(x.shape, axis, reduceType);
    const output = this.makeOutputArray(program.outputShape, 'int32') as Tensor;
    return this.compileAndRun(program, [x], output, [axis]) as Tensor;
  }

  argMin(x: Tensor, axis: number): Tensor {
    return this.argMinMaxReduce(x, axis, 'min');
  }

  argMax(x: Tensor, axis: number): Tensor {
    return this.argMinMaxReduce(x, axis, 'max');
  }

  concat(tensors: Tensor[], axis: number): Tensor {
    if (tensors.length === 1) {
      return tensors[0];
    }
    const outShape = computeOutShape(tensors.map(t => t.shape), axis);
    const tensors2D = tensors.map(t => t.reshape([
      util.sizeFromShape(t.shape.slice(0, axis)),
      util.sizeFromShape(t.shape.slice(axis))
    ]) as Tensor2D);
    const program = new ConcatProgram(tensors2D.map(t => t.shape));
    const res = this.compileAndRun(program, tensors2D) as Tensor;
    const result = res.reshape(outShape);
    return result;
  }

  maxPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    const program = new MaxPoolProgram(convInfo);

    const output =
        this.makeOutputArray(program.outputShape, x.dtype) as Tensor4D;

    const dimensions = [
      convInfo.padInfo.left, convInfo.padInfo.top,      // Padding.
      convInfo.strideWidth, convInfo.strideHeight,      // Stride.
      convInfo.dilationWidth, convInfo.dilationHeight,  // Dilation.
      convInfo.inWidth, convInfo.inHeight,              // Conv dims.
      convInfo.effectiveFilterWidth,
      convInfo.effectiveFilterHeight  // Filter dims.
    ];

    return this.compileAndRun(program, [x], output, dimensions);
  }

  dispose() {
    // Backend disposal logic.
  }
}
