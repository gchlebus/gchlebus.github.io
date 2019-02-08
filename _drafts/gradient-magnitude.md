---
layout: post
title: "Gradient magnitude is what acually matters"
excerpt: ""
categories: neural networks
---

I did an interesting observation when training a 3D neural network for segmentation of livers in CT
scans. First, I trained a model by only showing patches containing both fg and bg voxels. This
training strategy led in some cases to failures, e.g., the produced liver mask leaked into
neighboring organs or even in some CTs legs were segmented instead of the liver. This phenomenon
made me think, that these problems could be mindered by training the model also with bg patches.
Counter-intuitevely, this strategy led to a substantial performance drop. So I started to
investigate what went wrong...

My set up was the following: U-net with 4 resolution levels, 2 output channels, softmax as the last
activation function, mini-batch size of 1 and dice loss function ignoring the background channel.

My first suspicion was that my model did not learn from the background patches. To verify this, I
recorded gradient norms over 1k iterations with fg+bg patches and bg patches only.

 - dice, fg+bg patches, average grad norm = 1.7
 - dice, bg patches, average grad norm = 0.00015

Gradients coming from bg patches were smaller by 4 orders of magnitude compared to fg+bg patches.
This means that the bg patches had a very small influence on model parameters updates compared to
fg+bg patches.

I investigated further how the gradient norm behaves when I use other loss functions:

- dice computed on all channels, fg+bg patches, average grad norm = 0.89
- dice computed on all channels, bg patches, average grad norm = 0.075
- categorical cross entropy, fg+bg patches, average grad norm = 1.47
- categorical cross entropy, bg patches, average grad norm = 0.26
- top-25% loss, fg+bg patches, average grad norm = 2.9
- top-25% loss, bg patches, average grad norm = 0.14

In case of other loss functions, the gradient magnitude coming from bg patches was smaller by a
factor of 10 than the gradient induced by fg+bg patches. This would mean, that these loss functions
can learn more from bg patches compared to the dice computed only on the fg output channel.