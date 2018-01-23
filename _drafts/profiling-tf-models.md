---
layout: post
title: "Profiling tensorflow models"
excerpt: "Check time and memory requirements of your tf models."
categories: neural networks
---

### Compile tf profiles
1. Install `bazel`in version `>= 0.5.4`.
2. `git clone tensorflow`
3. `cd tensorflow`
4. `./configure` (I chose no CUDA support)
5. ` bazel build --config opt tensorflow/core/profiler:profiler`

TODO:
- Memory consumption
- time requirements
- weight count
- use package from openai / experiment with custom checkpoints
- change to tf.float16
- batch size vs memory consumption
- from tensorflow.python.client import timeline
- profiling with tensorboard
