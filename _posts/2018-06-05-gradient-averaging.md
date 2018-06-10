---
layout: post
title: "Gradient averaging with TensorFlow"
excerpt: "Gradient averaging over multiple training steps is a very useful technique, which can help you overcome the limitations of your GPU."
categories: neural networks
date: 2018-06-05
comments: true
---

When training big neural networks, it can happen that the biggest mini-batch size you can afford is one.
In such cases training can get very inefficient and even not converge due to very noisy gradients.
Gradient averaging is a technique allowing to increase the effective mini-batch size arbitralily despite GPU memory constraints.
The key idea is to separate gradients computation from applying them.
If you do so, you can compute gradients in each iteration and apply an average of them less frequently.
Let's take a look at a code examples (full code can be found here[^1]).

**Separate gradient computation**

```python
def setup_train(self, average_gradients=1, lr=1e-3):
    self._average_gradients = average_gradients
    self._loss_op = tf.losses.softmax_cross_entropy(onehot_labels=self._labels, logits=self._inference_op)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    if average_gradients == 1:
      # This 'train_op' computes gradients and applies them in one step.
      self._train_op = optimizer.minimize(self._loss_op)
    else:
      # here 'train_op' only applies gradients passed via placeholders stored
      # in 'grads_placeholders. The gradient computation is done with 'grad_op'.
      grads_and_vars = optimizer.compute_gradients(self._loss_op)
      avg_grads_and_vars = []
      self._grad_placeholders = []
      for grad, var in grads_and_vars:
        grad_ph = tf.placeholder(grad.dtype, grad.shape)
        self._grad_placeholders.append(grad_ph)
        avg_grads_and_vars.append((grad_ph, var))
      self._grad_op = [x[0] for x in grads_and_vars]
      self._train_op = optimizer.apply_gradients(avg_grads_and_vars)
      self._gradients = [] # list to store gradients
```

**Train step**
```python
def train(self, session, input_batch, output_batch):
    feed_dict = {
      self._input: input_batch,
      self._labels: output_batch,
      self._training: True
    }
    if self._average_gradients == 1:
      loss, _ = session.run([self._loss_op, self._train_op], feed_dict=feed_dict)
    else:
      loss, grads = session.run([self._loss_op, self._grad_op], feed_dict=feed_dict)
      self._gradients.append(grads)
      if len(self._gradients) == self._average_gradients:
        for i, placeholder in enumerate(self._grad_placeholders):
          feed_dict[placeholder] = np.stack([g[i] for g in self._gradients], axis=0).mean(axis=0)
        session.run(self._train_op, feed_dict=feed_dict)
        self._gradients = []
    return loss
```

### Experiment

Let's see, how gradient averaging affects model performance.
For that, I trained the same network with different mini-batch sizes, but the gradients were applied always after the network has seen 100 samples.
Moreover, the maximum number of iterations was set such that in each case the same count of training steps was performed.

```
>>> python main.py --average-gradients=1 --batch-size=100 --iterations=1000
mean accuracy (10 runs): 95 +/- 0.172

>>> python main.py --average-gradients=5 --batch-size=20 --iterations=5000
mean accuracy (10 runs): 94.9 +/- 0.325

>>> python main.py --average-gradients=10 --batch-size=10 --iterations=10000
mean accuracy (10 runs): 95.1 +/- 0.36

>>> python main.py --average-gradients=20 --batch-size=5 --iterations=20000
mean accuracy (10 runs): 95.2 +/- 0.303
```

#### References
[^1]: [Gradient averaging code example](https://github.com/gchlebus/gchlebus.github.io/tree/master/code/gradient-averaging)