---
layout: post
title: "Profiling tensorflow models"
excerpt: "Check time and memory requirements of your tf models."
categories: neural networks
---

In this post I will show you how to get more insight into your model.

### Base architecure
U-net.

### Trainable parameter count
Once you have constructed your model, you can query the trainable parameter count using the following function:
```python
import tensorflow as tf
import numpy as np

def parameter_count(graph):
  vars = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  return (np.sum([np.prod(v.shape) for v in vars])).value
```

```
>>> from u_net import UNet
>>> u_net = UNet()
>>> parameter_count(tf.get_default_graph())
31030658
```

### Memory consumption
To check the model's memory consumption I will use the `mem_util.peak_memory` helper from the openai repository[^1]. I equipped the `UNet` class with `train` and `inference` methods accepting `RunOptions` and `RunMetadata`, which collect debug/profiling information from a `session.run()` call. The following function returns a dict containing per device peak memory consumption in bytes:
```python
import tensorflow as tf
import numpy as np
from u_net import UNet
from gradientcheckpointing.test import mem_util # from openai repo

def peak_memory(batch_size=1, mode='train'):
  u_net = UNet()
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  input_shape = [batch_size] + UNet.INPUT_SHAPE
  output_shape = [batch_size] + UNet.OUTPUT_SHAPE
  input_batch = np.random.rand(*input_shape)
  output_batch = np.random.rand(*output_shape)
  run_metadata = tf.RunMetadata()
  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  if mode == 'train':
    u_net.train(sess, input_batch, output_batch, options, run_metadata)
  elif mode == 'inference':
    u_net.inference(sess, input_batch, options, run_metadata)
  return mem_util.peak_memory(run_metadata)
```
```
 >>> peak_memory(mode='train')
{'/cpu:0': 2093226660}
>>> peak_memory(mode='inference')
{'/cpu:0': 180936736}
```
From the above we can see, that the U-Net model requires ca. 2 GB of memory in the train mode and only 180 MB in the inference mode. We can run the above code to obtain train peak memory for different batch sizes. From the below plot, we can clearly see, that the peak memory increases linearly with the batch size.
![Memory_BatchSizr]({{ "/assets/profiling-tf-models/memory_batchsize.png" | absolute_url }})

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

---
#### References
[^1]: [openai/gradient-checkpointing](https://github.com/openai/gradient-checkpointing)