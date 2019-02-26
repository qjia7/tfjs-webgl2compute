import * as tf from '@tensorflow/tfjs';

import {WebGPUKernelBackend} from './backend_webgpu';

export * from '@tensorflow/tfjs';

tf.ENV.registerBackend('tensorflow', () => {
  return new WebGPUKernelBackend();
}, 3 /*priority*/);

// If registration succeeded, set the backend.
if (tf.ENV.findBackend('tensorflow') != null) {
  tf.setBackend('tensorflow');
}