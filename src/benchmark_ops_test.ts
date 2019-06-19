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

import * as tf from '@tensorflow/tfjs-core';
import * as tfwebgl2compute from './index';
import {MobileNetV1GPUBenchmark} from './mobilenet_benchmarks';
import * as test_util from './test_util';

describe('Ops benchmarks', () => {
  beforeAll(async () => {
    await tfwebgl2compute.ready;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;
  });

    it('mobilenet_v1', async () => {
    const sizes = [1];  // MobileNet version
    const runs = 20;

    const benchmark = new MobileNetV1GPUBenchmark();
    await benchmark.loadModel();

    await test_util.benchmarkAndLog(
        'mobilenet_v1', size => benchmark.run(size), sizes,
        size => `N=${size}_0_224`, runs);
  });

  it('matMul', async () => {
    const times = [];

    const a = tf.randomNormal([500, 500]);
    const b = tf.randomNormal([500, 500]);

    let c = tf.matMul(a, b);
    await c.data();

    for (let i = 0; i < 100; i++) {
      const start = performance.now();
      c = tf.matMul(a, b);
      await c.data();
      times.push(performance.now() - start);
    }

    a.dispose();
    b.dispose();
    console.log(`MatMul: Average time ms: ${
        times.reduce((a, b) => a + b, 0) / times.length}`);
    console.log(`Min time ms: ${Math.min(...times)}`);
  });

  it('conv2d', async () => {
    const times = [];

    const a = tf.randomNormal<tf.Rank.R4>([1, 128, 128, 4]);
    const b = tf.randomNormal<tf.Rank.R4>([25, 25, 4, 4]);

    let c = tf.conv2d(a, b, 1, 'same');
    await c.data();

    for (let i = 0; i < 100; i++) {
      const start = performance.now();
      c = tf.conv2d(a, b, 1, 'same');
      await c.data();
      times.push(performance.now() - start);
    }

    a.dispose();
    b.dispose();
    console.log(`Conv2d: Average time ms: ${
        times.reduce((a, b) => a + b, 0) / times.length}`);
    console.log(`Min time ms: ${Math.min(...times)}`);
  });
});