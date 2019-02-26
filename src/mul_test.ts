import * as tf from './index';
import {expectArraysClose} from '@tensorflow/tfjs-core/dist/test_util';

describe("WebGPU backend", () => {
  it("A * B elementwise", () => {
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.tensor1d([3, 4, 5]);
    const c = tf.mul(a, b);

    expectArraysClose(c, [3, 8, 15]);
  });
});