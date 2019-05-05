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

import {DataMover, DataType, KernelBackend, Rank, ShapeMap, Tensor, Tensor3D, util} from '@tensorflow/tfjs-core';

import * as binary_op from './kernels/binary_op';
import {BinaryOpProgram} from './kernels/binary_op';
import {MatMulProgram} from './kernels/matmul';
import * as unary_op from './kernels/unary_op';
import {UnaryOpProgram} from './kernels/unary_op';
import * as webgl2compute_math from './kernels/webgl2compute_math';

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

  floatPrecision(): number {
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
      output?: Tensor): K {
    if (output == null) {
      output = this.makeOutputArray(program.outputShape, inputs[0].dtype);
    }

    const key = webgl2compute_math.makeShaderKey(program);
    const binary = this.getAndSaveBinary(key, () => {
      return webgl2compute_math.compileProgram(program, this.gl);
    });

    this.gl.useProgram(binary);
    // TODO: Add uniform support.
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

    return output as {} as K;
  }

  add(a: Tensor, b: Tensor): Tensor {
    const output = Tensor.make(a.shape, {}, a.dtype, this);
    const program = new BinaryOpProgram(binary_op.ADD, output.shape);

    return this.compileAndRun(program, [a, b], output) as Tensor;
  }

  multiply(a: Tensor, b: Tensor): Tensor {
    const output = Tensor.make(a.shape, {}, a.dtype, this);
    const program = new BinaryOpProgram(binary_op.MUL, output.shape);

    return this.compileAndRun(program, [a, b], output) as Tensor;
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
    const sharedDim = transposeA ? a.shape[1] : a.shape[2];
    const [batch, , ] = a.shape;

    const output =
        Tensor.make([batch, outerShapeA, outerShapeB], {}, a.dtype, this) as
        Tensor3D;

    const program = new MatMulProgram(
        output.shape, [outerShapeA, sharedDim, outerShapeB, batch]);

    return this.compileAndRun(program, [a, b], output) as Tensor3D;
  }
}
