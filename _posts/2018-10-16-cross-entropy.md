---
layout: post
title: "Cross-entropy based loss functions"
excerpt: "Differences between categorical cross-entropy and binary cross-entropy loss functions."
categories: neural networks
date: 2018-10-16
comments: true
---

Binary cross-entropy and categorical cross-entropy are two most common cross-entropy based
loss function, that are available in deep learning frameworks like Keras. For a
classification problem with $$N$$ classes the cross-entropy $$\textrm{CE}$$ is defined:

$$\textrm{CE} = -\sum_{i}^{N}p_i \log{q_i}$$

Where $$p_i$$ denotes whether the input belongs to the class $$i$$ and $$q_i$$ is the
predicted score for class $$i$$.

### Categorical cross-entropy
Categorical cross-entropy $$\textrm{CCE}$$ is $$\textrm{CE}$$ where the vector $$q$$ is
computed using softmax function. Softmax squashes the input vector into a vector which
represents a valid probability distribution (i.e. sums up to 1). $$\textrm{CCE}$$ is
suitable for multi-class problems, where given input can belong only to one class (classes
are mutually exclusive). $$\textrm{CCE}$$ can be implemented in the following way:
```python
def cce_loss(softmax_output, target):
  softmax_output = np.asfarray(softmax_output)
  target = np.asfarray(target)
  return -target * np.log(softmax_output)
```

#### Example
```bash
>>> target = [0, 1, 0]
>>> softmax_output = [0.3, 0.6, 0.1]
>>> print cce_loss(softmax_output, target)
[0, 0.5108256, 0]
```
$$\textrm{CE} = -\sum_{i}^{N}p_i \log{q_i} = -(0 \cdot \log{0.3} + 1 \cdot \log{0.6} + 0 \cdot \log{0.1}) = 0.5108625$$

From the above we see, that only the probability of the true class contributes to the
total loss. Model outputs for the other classes are influenced indirectly by the softmax
activation, which works according to the "winner takes it all" principle.

### Binary cross-entropy
The binary cross-entropy $$\textrm{BCE}$$ function considers each class score produced by
the model independently, which makes this loss function suitable also for multi-label
problems, where each input can belong to more than one class. Unlike $$\textrm{CCE}$$,
$$\textrm{BCE}$$ doesn't assume a specific activation function of the final network layer.
For a problem with 3 output classes (A, B, C) the binary cross-entropy considers three
independent binary classification problems:

- class A vs. not class A
- class B vs. not class B
- class C vs. not class C

$$\textrm{BCE}$$ is defined as:

$$\textrm{BCE} = \sum_{i}^{N}(-\sum_{j}^{N'=2}p_j \log{q_j}) = -\sum_{i}^{N}(p_i \log{q_i} + (1-p_i)\log{(1-q_i)}) $$

Python implementation
```python
def bce_loss(output, target):
  output = np.asfarray(output)
  target = np.asfarray(target)
  d = np.abs(output - target)
  return -np.log(1 - d)
```

#### Example
```bash
>>> target = [0, 1, 0]
>>> output = [0.3, 0.6, 0.1]
>>> print bce_loss(p, q)
[0.35667497 0.5108256  0.10536055]
```
$$\textrm{BCE} = -( (1-0)\cdot\log{(1-0.3)} + 1\cdot\log{0.6} + (1-0)\log{(1-0.1)} ) = $$
$$= 0.35667497 + 0.5108256 + 0.10536055$$

