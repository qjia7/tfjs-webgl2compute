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

import './flags';

import {backend_util, DataMover, DataType, ENV, KernelBackend, Rank, ShapeMap, Tensor, Tensor2D, Tensor3D, Tensor4D, util} from '@tensorflow/tfjs-core';

import {ArgMinMaxProgram} from './kernels/argminmax';
import * as binary_op from './kernels/binary_op';
import {BinaryOpProgram} from './kernels/binary_op';
import {ConcatProgram} from './kernels/concat';
import {Conv2DMMProgram} from './kernels/conv2d_mm';
import {Conv2DNaiveProgram} from './kernels/conv2d_naive';
import {DepthwiseConv2DProgram} from './kernels/conv_gpu_depthwise';
import {MatMulProgram} from './kernels/matmul';
import {MatMul8x8And4x16Program} from './kernels/matmul_8x8_4x16';
import {MatMulPackedProgram} from './kernels/matmul_packed';
import {MaxPoolProgram} from './kernels/maxpool';
import {PadProgram} from './kernels/pad';
import {ResizeBilinearProgram} from './kernels/resize_bilinear';
import {TransposeProgram} from './kernels/transpose';
import * as unary_op from './kernels/unary_op';
import {UnaryOpProgram} from './kernels/unary_op';
import * as webgl2compute_math from './kernels/webgl2compute_program';

type TensorInfo = {
  byteSize: number,
  values: Float32Array|Int32Array|Uint8Array,
  id: number,
  dtype: DataType,
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
  private fromPixels2DContext: CanvasRenderingContext2D;

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
      const byteSize = util.sizeFromShape(shape) * util.bytesPerElement(dtype);
      const buffer = this.gl.createBuffer();
      // tslint:disable-next-line:no-any
      this.gl.bindBuffer((this.gl as any).SHADER_STORAGE_BUFFER, buffer);
      this.gl.bufferData(
          // tslint:disable-next-line: no-any
          (this.gl as any).SHADER_STORAGE_BUFFER, byteSize,
          this.gl.STATIC_DRAW);
      this.tensorMap.set(
          dataId, {byteSize, values: null, id: -1, buffer, dtype});
    }
  }

  write(dataId: object, values: Float32Array|Int32Array|Uint8Array): void {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }

    const info = this.tensorMap.get(dataId);
    info.values = values;
    // tslint:disable-next-line: no-any
    this.gl.bindBuffer((this.gl as any).SHADER_STORAGE_BUFFER, info.buffer);
    // tslint:disable-next-line: no-any
    this.gl.bufferSubData((this.gl as any).SHADER_STORAGE_BUFFER, 0, values);
  }

  async read(dataId: object): Promise<Float32Array|Int32Array|Uint8Array> {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }
    const info = this.tensorMap.get(dataId);
    const size = info.byteSize;
    // tslint:disable-next-line:no-any
    this.gl.bindBuffer((this.gl as any).SHADER_STORAGE_BUFFER, info.buffer);
    const data = info.dtype === 'int32' ? new Int32Array(size / 4) :
                                          new Float32Array(size / 4);
    // tslint:disable-next-line: no-any
    (this.gl as any)
        // tslint:disable-next-line: no-any
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
    let currentOffset = 0;
    bufferShapes.forEach((d, i) => {
      // Uniforms.
      if (d.length === 0) {
        d = [1];
      }
      // Complete std140 layout rules are documented here:
      // tslint:disable-next-line:max-line-length
      // https://www.khronos.org/registry/OpenGL/specs/gl/glspec45.core.pdf#page=159
      let baseAlignment: number;
      switch (d.length) {
        case 0:
          baseAlignment = 1;
          break;
        case 1:
          baseAlignment = 1;
          break;
        case 2:
          baseAlignment = 2;
          break;
        case 3:
          baseAlignment = 4;
          break;
        case 4:
          baseAlignment = 4;
          break;
        default:
          util.assert(false, () => `Unsupported ${d.length}D shape`);
      }

      const padding = Math.ceil(currentOffset / baseAlignment) * baseAlignment -
          currentOffset;
      for (let p = 0; p < padding; ++p) {
        dimUniforms.push(0);
      }
      dimUniforms.push(...d);
      currentOffset += d.length + padding;
    });

    if (programUniforms) {
      dimUniforms = dimUniforms.concat(programUniforms);
    }

    const key = webgl2compute_math.makeShaderKey(
        program, bufferShapes.map(d => d.length));
    const inputsData =
        inputs.map((input: Tensor, i: number) => ({
                     // Returning dtype from tensorMap because it reflects dtype
                     // of underlying buffer, rather than abstract dtype.
                     dtype: this.tensorMap.get(input.dataId).dtype,
                     shape: input.shape,
                     name: program.variableNames[i]
                   }));
    const binary = this.getAndSaveBinary(key, () => {
      return webgl2compute_math.compileProgram(
          program, inputsData, output, this.gl);
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
      // tslint:disable-next-line: no-any
      (this.gl as any)
          .bindBufferBase(
              // tslint:disable-next-line: no-any
              (this.gl as any).SHADER_STORAGE_BUFFER, i, mapInfo.buffer);
      outputBinding = outputBinding + 1;
    });

    const mapInfo = this.tensorMap.get(output.dataId);
    // tslint:disable-next-line: no-any
    (this.gl as any)
        .bindBufferBase(
            // tslint:disable-next-line: no-any
            (this.gl as any).SHADER_STORAGE_BUFFER, outputBinding,
            mapInfo.buffer);

    // tslint:disable-next-line: no-any
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
        // tslint:disable-next-line: no-any
        (this.gl as any).UNIFORM_BUFFER, data, this.gl.STATIC_DRAW);
    // tslint:disable-next-line: no-any
    (this.gl as any).bindBufferBase((this.gl as any).UNIFORM_BUFFER, 0, buffer);
    return buffer;
  }

  private binaryOp(a: Tensor, b: Tensor, op: string) {
    const dtype = backend_util.upcastType(a.dtype, b.dtype);
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

  resizeBilinear(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    const program =
        new ResizeBilinearProgram(x.shape, newHeight, newWidth, alignCorners);

    const output =
        this.makeOutputArray(program.outputShape, x.dtype) as Tensor4D;

    return this.compileAndRun(program, [x], output) as Tensor4D;
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

    let program: MatMulProgram|MatMulPackedProgram|MatMul8x8And4x16Program;
    // TODO: We should eventually use the blocked version, but keeping around
    // the old version while we try to understand conditions under which blocked
    // is faster.
    if (ENV.get('MATMUL_WORK_PER_THREAD') === 0) {
      program = new MatMulProgram(output.shape);
    } else if (a.shape[2] % 4 === 0 && b.shape[2] % 4 === 0) {
      // TODO: Currently we need to make sure that a.shape[2] and b.shape[2] are
      // divided by 4 since we use vec4 to get data. In future, we can remove
      // this limitation by insert 0 to pack data.
      program = new MatMul8x8And4x16Program(output.shape);
    } else {
      program = new MatMulPackedProgram(
          output.shape, ENV.get('MATMUL_WORK_PER_THREAD') as number);
    }

    return this.compileAndRun(program, [a, b], output) as Tensor3D;
  }

  conv2d(x: Tensor4D, filter: Tensor4D,
        convInfo: backend_util.Conv2DInfo): Tensor4D {
    const output =
        Tensor.make(convInfo.outShape, {}, x.dtype, this) as Tensor4D;
    let program: Conv2DNaiveProgram|Conv2DMMProgram;
    const workPerThread = ENV.get('CONV2D_WORK_PER_THREAD') as number;
    if (workPerThread === -1) {
      // TODO(kainino0x): This may be obsolete, but is kept for reference.
      program = new Conv2DNaiveProgram(convInfo);
    } else {
      program = new Conv2DMMProgram(convInfo, workPerThread);
    }

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

  depthwiseConv2D(x: Tensor4D, filter: Tensor4D, convInfo: backend_util.Conv2DInfo):
      Tensor4D {
    let program: DepthwiseConv2DProgram;
    program = new DepthwiseConv2DProgram(convInfo);
    return this.compileAndRun(program, [x, filter]);
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
    const outShape = backend_util.computeOutShape(
                         tensors.map(t => t.shape), axis);
    const tensors2D = tensors.map(t => t.reshape([
      util.sizeFromShape(t.shape.slice(0, axis)),
      util.sizeFromShape(t.shape.slice(axis))
    ]) as Tensor2D);
    const program = new ConcatProgram(tensors2D.map(t => t.shape));
    const res = this.compileAndRun(program, tensors2D) as Tensor;
    const result = res.reshape(outShape);
    return result;
  }

  maxPool(x: Tensor4D, convInfo: backend_util.Conv2DInfo): Tensor4D {
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

  pad<T extends Tensor>(
      x: T, paddings: Array<[number, number]>, constantValue: number): T {
    const program = new PadProgram(x.shape, paddings, constantValue);
    const output = this.makeOutputArray(program.outputShape, x.dtype);
    return this.compileAndRun(program, [x], output);
  }

  fromPixels(
      pixels: backend_util.PixelData|ImageData|HTMLImageElement|
      HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): Tensor3D {
    if (pixels == null) {
      throw new Error(
          'pixels passed to tf.browser.fromPixels() can not be null');
    }

    const outShape = [pixels.height, pixels.width, numChannels];
    let imageData = (pixels as ImageData | backend_util.PixelData).data;

    if (ENV.getBool('IS_BROWSER')) {
      if (!(pixels instanceof HTMLVideoElement) &&
          !(pixels instanceof HTMLImageElement) &&
          !(pixels instanceof HTMLCanvasElement) &&
          !(pixels instanceof ImageData) &&
          !((pixels as backend_util.PixelData).data instanceof Uint8Array)) {
        throw new Error(
            'pixels passed to tf.browser.fromPixels() must be either an ' +
            `HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData` +
            ` or {data: Uint32Array, width: number, height: number}, ` +
            `but was ${(pixels as {}).constructor.name}`);
      }
      if (pixels instanceof HTMLVideoElement) {
        if (this.fromPixels2DContext == null) {
          this.fromPixels2DContext =
              document.createElement('canvas').getContext('2d');
          this.fromPixels2DContext.canvas.width = pixels.width;
          this.fromPixels2DContext.canvas.height = pixels.height;
        }
        this.fromPixels2DContext.drawImage(
            pixels, 0, 0, pixels.width, pixels.height);
        pixels = this.fromPixels2DContext.canvas;
      }

      // TODO: Workround to upload and encoding textures on GPU, will
      // follow WebGL's solution
      const imageDataLivesOnGPU = pixels instanceof HTMLVideoElement ||
          pixels instanceof HTMLImageElement ||
          pixels instanceof HTMLCanvasElement;
      if (imageDataLivesOnGPU) {
        imageData = this.fromPixels2DContext
                        .getImageData(0, 0, pixels.width, pixels.height)
                        .data;
      }
    }

    let pixelArray = imageData;
    if (numChannels != null && numChannels !== 4) {
      pixelArray = new Uint8Array(pixels.width * pixels.height * numChannels);

      for (let i = 0; i < imageData.length; i++) {
        if (i % 4 < numChannels) {
          const pixelIndex = Math.floor(i / 4);
          pixelArray[pixelIndex * numChannels + i % 4] = imageData[i];
        }
      }
    }

    const output = this.makeOutputArray(outShape, 'int32');
    this.write(output.dataId, Int32Array.from(pixelArray));
    return output as Tensor3D;
  }

  cast<T extends Tensor>(x: T, dtype: DataType): T {
    return backend_util.castTensor(x, dtype, this);
  }

  dispose() {
    // Backend disposal logic.
  }
}
