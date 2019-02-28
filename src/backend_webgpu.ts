import {KernelBackend, DataMover, Tensor, DataType} from '@tensorflow/tfjs-core';

type TensorInfo = {
  shape: number[],
  dtype: DataType,
  values: Float32Array|Int32Array|Uint8Array,
  id: number
};

interface DataId {}

export class WebGPUBackend extends KernelBackend {
  floatPrecision(): number {
    return 32;
  }

  setDataMover(dataMover: DataMover): void {
    // TODO: tfjs team to implement this.
  }

  private tensorMap = new WeakMap<DataId, TensorInfo>();

  disposeData(dataId: DataId): void {
    // Tensor disposal logic.
  }

  register(dataId: object, shape: number[], dtype: DataType) {
    if (!this.tensorMap.has(dataId)) {
      this.tensorMap.set(
          dataId, {shape, dtype, values: null, id: -1});
    }
  }

  write(dataId: object, values: Float32Array|Int32Array|Uint8Array): void {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }

    const info = this.tensorMap.get(dataId);
    info.values = values;
    this.tensorMap.set(dataId, info);
  }

  async read(dataId: object): Promise<Float32Array|Int32Array|Uint8Array> {
    const data = await new Float32Array([0, 0, 0]);
    return data;
  }

  multiply(a: Tensor, b: Tensor) {
    // Lazily create / upload textures and execute WebGPU kernel.
    return a;
  }
}