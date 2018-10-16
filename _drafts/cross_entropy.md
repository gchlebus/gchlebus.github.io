---
layout: post
title: "Cross-Entropy loss"
excerpt: "Some considerations about the cross-entropy loss."
categories: neural networks
---

Cross-entropy loss for one pixel is defined as:

$$\textrm{CE} = -\sum_{i}^{N}p_i \log{q_i}$$

Where $$N$$ denotes the number of classes, $$p_i$$ whether the pixel belongs to the class
$$i$$ and $$q_i$$ is the model score.

### Categorical cross-entropy loss
This is basically softmax activation function + cross-entropy. This loss function is
applied to multi-class problems, where a particular pixel can belong only to one class
(classes are mutually exclusive).

#### Example
```
>>> p = [0, 1, 0]
>>> q = [0.3, 0.6, 0.1]
>>> print cce_loss(p, q)
[0, 0.5108256, 0]
```
$$\textrm{CE} = -\sum_{i}^{N}p_i \log{q_i} = -(0 \cdot \log{0.3} + 1 \cdot \log{0.6} + 0 \cdot \log{0.1}) = 0.5108625$$

From the above we see, that only the probability of the true class contributes to the
total loss. Model outputs for the other classes are influenced indirectly by the softmax
activation, which works according to the "winner takes it all" principle.

### Binary cross-entropy loss
The binary cross-entropy loss considers each class score produced by the model
independently, which makes this loss function suitable for multi-label problems, where
each pixel can belong to more than one class. For a problem with 3 output classes (A, B, C) the
binary cross-entropy considers three independent binary classification problems:

- class A vs. not class A
- class B vs. not class B
- class C vs. not class C

The binary cross-entropy is defined as:

$$\textrm{BCE} = \sum_{i}^{N}(-\sum_{j}^{N'=2}p_j \log{q_j}) = -\sum_{i}^{N}(p_i \log{q_i} + (1-p_i)\log{(1-q_i)}) $$


#### Example
```
>>> p = [0, 1, 0]
>>> q = [0.3, 0.6, 0.1]
>>> print bce_loss(p, q)
[0.35667497 0.5108256  0.10536055]
```
