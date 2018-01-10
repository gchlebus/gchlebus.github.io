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

$$y_i^{l} = f(\textbf{w}_i \textbf{y}^{l-1} + b_i)$$


[^1]: Srivastava at al., [*Dropout: A Simple Way to Prevent Neural Networks from Overfitting*](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf). Journal of Machine Learning Research, 2014.