---
layout: post
title: Dropout
categories: neural network
---

Dropout[^1] is a very popular regularization technique which can be injected into most of the neural network architectures. Together with other methods such as L1-/L2-norm regularization, soft weight sharing it helps deep neural nets in fighting overfitting.

### Motivation
The inspiration of the dropout is rooted in the sexual reproduction, which is how the most intelligent animals evolved. An offspring conceived in such a way inherits genes from both of parents. It seems counter-intuitive, that this way of reproduction was chosen over asexual one, where one parent passes all of his genes with some mutation allowing for further boost of individual fitness. One explanation for this phenomenon is that what really matters is not the individual fitness, but the ability of genes to do something useful when mixed with a new random set of genes. It forces genes not to depend too much on the presence of specific genes. Dropout aims at achieving the same behavior for neurons.

### Method
Before describing how we can incorporate dropout into a neural net, we need to define what it means to *drop* a neural unit. Dropping a unit with a drop probability $$p_d$$ corresponds to removing the neuron from the network with all its input and output connections with a probability $$p_d$$. The authors claim, that a quite optimal dropout probability is $$0.5$$ for hidden layers and $$0.2$$ for input layers.

We can write the following equation for output $$y_i$$ of a neuron unit $$i$$ in layer $$l$$ having weights $$\textbf{w}_i$$ and bias $$b_i$$. $$\textbf{y}^{l-1}$$ denotes the input vector.

1. $$y_i^{l} = f(\textbf{w}_i \textbf{y}^{l-1} + b_i)$$

In training, inserting dropout between layers $$l-1$$ and $$l$$ would alter the eq. 1 in the following way:

$$r_i^{l-1} \sim \operatorname{Bernoulli}(p_d)$$

$$y_i^{l} = f(\textbf{w}_i (\textbf{r}^{l-1} \cdot \textbf{y}^{l-1}) + b_i)$$

The Bernoulli distribution is of a random variable which takes value $$0$$ with probability $$p_d$$, "$$\cdot$$" denotes a elementwise multiplication. To account for the lower expected value of activities coming out of the $$l$$ layer in the training phase, we need to scale the weights $$\textbf{w}$$ at the inference time:

$$\textbf{w}_{\textrm{inference}} = (1-p_d)\textbf{w}_{\textrm{training}}$$

### Remarks
Imposing an upper bound on the weights from the incoming layer was found to be particularly beneficial for, but not only, dropout networks. This is a so-called max-norm regularization, where the weight optimization is subject to $$\|\textbf{w}\|_2 \lt c $$. It allows for using of bigger learning rates without the possibility of weights exploding.

### Experiment
I implemented a LeNet-like model and run the following experiments. Please check [this code](https://github.com/gchlebus/gchlebus.github.io/blob/master/code/dropout/dropout_experiment.py) to learn about the exact model implementation.

```
>>> python dropout_experiment.py
mean accuracy (10 runs): 73.1 +/- 14
>>> python dropout_experiment.py --dropout
mean accuracy (10 runs): 97.9 +/. 0.107
>>> python dropout_experiment.py --dropout --max_norm 10
mean accuracy (10 runs): 97.7 +/. 0.0973
```

---
#### References
[^1]: Srivastava at al., [*Dropout: A Simple Way to Prevent Neural Networks from Overfitting*](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf). Journal of Machine Learning Research, 2014.