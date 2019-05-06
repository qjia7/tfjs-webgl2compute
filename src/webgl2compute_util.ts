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

import {util} from '@tensorflow/tfjs-core';

// TODO: Figure out what kind of workgroup size is best?
export function computeWorkGroupSize(outputShape: number[]):
    [number, number, number] {
  const size = util.sizeFromShape(outputShape);
  let x = 16;
  if (size > 512) {
    x = 512;
  }
  if (size > 256) {
    x = 256;
  }
  if (size > 128) {
    x = 128;
  }
  if (size > 64) {
    x = 64;
  }
  if (size > 32) {
    x = 32;
  }
  if (size > 16) {
    x = 16;
  }
  return [x, 1, 1];
}
