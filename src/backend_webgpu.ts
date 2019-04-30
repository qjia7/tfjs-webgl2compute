import {KernelBackend, DataMover, Tensor, DataType, util} from '@tensorflow/tfjs-core';

import {MultiplyProgram} from './kernels/multiply';
import * as webgpu_math from './kernels/webgl2compute_math';

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

export class WebGPUBackend extends KernelBackend {
  gl: WebGLRenderingContext;
  private tensorMap = new WeakMap<DataId, TensorInfo>();

  constructor() {
      super();
      this.gl = getWebGLContext();
  }

  floatPrecision(): number {
    return 32;
  }

  setDataMover(dataMover: DataMover): void {
    // TODO: tfjs team to implement this.
  }

  disposeData(dataId: DataId): void {
    // Tensor disposal logic.
  }

  register(dataId: object, shape: number[], dtype: DataType): void {
    if (!this.tensorMap.has(dataId)) {
      const buffer = this.gl.createBuffer();
      // tslint:disable-next-line:no-any
      this.gl.bindBuffer((this.gl as any).SHADER_STORAGE_BUFFER, buffer)
      this.gl.bufferData((this.gl as any).SHADER_STORAGE_BUFFER,
                         util.sizeFromShape(shape) * util.bytesPerElement(dtype),
                         this.gl.STATIC_DRAW);
      this.tensorMap.set(
          dataId, {shape, dtype, values: null, id: -1, buffer});
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
    const size = util.sizeFromShape(info.shape) * util.bytesPerElement(info.dtype);
    // tslint:disable-next-line:no-any
    this.gl.bindBuffer((this.gl as any).SHADER_STORAGE_BUFFER, info.buffer);
    const data = new Float32Array(size/4);
    (this.gl as any).getBufferSubData((this.gl as any).SHADER_STORAGE_BUFFER, 0, data);

    return data;
  }


  private makeOutputArray<T extends Tensor>(shape: number[], dtype: DataType):
    T {
    return Tensor.make(shape, {}, dtype, this) as T;
  }

  private compileAndRun<
      K extends {dtype: DataType, size: number, dataId: {}, shape: number[]}>(
      program: webgpu_math.WebGL2ComputeProgram, inputs: Tensor[], output?: Tensor): K {
    if (output == null) {
      output = this.makeOutputArray(program.outputShape, inputs[0].dtype);
    }

    // TODO: Cache the compile binary.
    webgpu_math.compileProgram(program, this.gl);

    let outputBinding = 0;
    inputs.forEach((input, i) => {
      const mapInfo = this.tensorMap.get(input.dataId);
      (this.gl as any).bindBufferBase((this.gl as any).SHADER_STORAGE_BUFFER, i, mapInfo.buffer);
      outputBinding = outputBinding + 1;
    });

    const mapInfo = this.tensorMap.get(output.dataId);
    (this.gl as any).bindBufferBase((this.gl as any).SHADER_STORAGE_BUFFER, outputBinding, mapInfo.buffer);

    (this.gl as any).dispatchCompute(program.dispatch[0], program.dispatch[1], program.dispatch[2]);

    return output as {} as K;
  }

  multiply(a: Tensor, b: Tensor): Tensor {
    const output = Tensor.make(a.shape, {}, a.dtype, this);
    const program = new MultiplyProgram(output.shape);
     return this.compileAndRun(program, [a, b], output) as Tensor;
  }

}