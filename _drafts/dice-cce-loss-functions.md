---
layout: post
title: "Dice vs. Categorical Cross Entropy"
excerpt: "See how dice and categorical cross entropy loss functions perform when training a semantic segmentation model."
categories: neural networks tensorflow
---

### Introduction
The categorical cross entropy (CCE) and The Dice index are popular loss functions for training of neural networks for semantic segmentation. In medical field images being analysed consist mainly of background pixels with few pixels belonging to objects of interest. Such cases of high class imbalance cause networks to be biased towards background when trained with CCE. To account for that, weighting of foreground and background pixels can be applied. In contrast to CCE, usage of the dice loss doesn't require weighting to successfully train models with imbalanced datasets[^1].

### Loss functions

#### Notation
In the following, $$y_i^j$$ and $$\hat{y}_i^j$$ denote $$i$$th channel of the $$j$$th pixel of the reference labels and neural network softmax output, respectively. I use $$c$$ to denote the total channel count, $$N$$ to denote total pixel count in a mini-batch and $$\epsilon$$ as a small constant plugged to avoid numerical problems. The code examples use `tensorflow` and assume that models are fed with 4-D tensors of shape `(batch_dim, y_dim, x_dim, channel_dim)`.

#### Categorical Cross Entropy

$$\textrm{cce} = -\sum_i^c \sum_j^N y_i^j\ln{\hat{y}_i^j}$$

```python
def cce_loss(softmax_output, labels):
    eps = 1e-7
    log_p = -tf.log(tf.clip_by_value(softmax_output, eps, 1-eps))
    loss = tf.reduce_sum(labels * log_p, axis=-1)
    return tf.reduce_mean(loss)
```

#### Dice
The Dice loss, as argued by Milletari[^1], requires no class-balancing to achieve a good segmentation quality on imbalanced datasets. The Dice loss function is defined:

$$\textrm{dice\_loss} = 1 - \frac{1}{c}\sum_{i=0}^{c}\frac{\sum_j^N 2y_i^j\hat{y}_i^j + \epsilon}{\sum_j^Ny_i^j + \sum_j^N\hat{y}_i^j + \epsilon}$$

$$\epsilon$$ is used to avoid division by 0 (denominator) and to learn from patches containing no labels in the reference (nominator). The multiplication by $$\frac{1}{c}$$ gives a nice property, that the loss is within $$[0, 1]$$ regardless of the channel count.

```python
def dice_loss(softmax_output, labels):
    axis = (0,1,2)
    eps = 1e-7
    nom = (2 * tf.reduce_sum(labels * softmax_output, axis=axis) + eps)
    denom = tf.reduce_sum(labels, axis=axis) + tf.reduce_sum(softmax_output, axis=axis) + eps
    return 1 - tf.reduce_mean(nom / denom)
```

---
#### References
[^1]: Milletari at al., [*V-Net: Fully Convolutional Neural Network for Volumetric Medical Image Segmentation*](https://arxiv.org/abs/1606.04797). 2016.
