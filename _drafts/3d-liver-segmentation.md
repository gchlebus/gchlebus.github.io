---
layout: post
title: "3D liver segmentation"
excerpt: ""
categories: neural networks segmentation
---

In the 3D U-net paper[^1] was shown, that a 3D convolutional neural network can be successfully trained for the task of semantic segmentation. The performance of a 3D network should be better than of a corresponding in terms of layer count 2D network, since the network can extract informations from a 3D context. The running example here would be liver segmentation in CT images. The challenge of 3D model training is that they have significantly bigger memory consumption, which means that one either has to reduce minibatch size or/and patch size.

The experiments here were run on GeForce GTX 1080 with 8GB RAM.

### Sanity checks
In order to check, whether a given network is cabaple of solving the task, one tries to train the model on only one example and see whether the network overfits. If it is the case, then test passed, it seems that the model has enough capacity to solve the problem.

In the following the 3D network is trained with `patch_size = (60, 52, 52)`, `padding = (44,)*3` and `minibatch_size = 1`. In order to put our expectation of a 3D model, let's start with a 2D network trained with similar settings.

#### 2D U-net + dropout + batch_renorm + cce loss
- `patch_size = (60, 60)`
- `padding = (44, 44)`
- `minibatch_size = 1`
- `learning_rage = 5e-4`
- train only with patches containing liver

The loss goes down to below 0.1, which seems fine.
![2DUnet_drop_bn_dice]({{ "/assets/3d-liver-segmentation/2DUnet_bn_drop_pb_dice.png" | absolute_url }})

#### 3D U-net + dropout + batch_renorm + dice loss
- `patch_size = (60, 60, 52)`
- `padding = (44, 44, 44)`
- `minibatch_size = 1`
- `learning_rage = 5e-4`
- train only with patches containing liver
![3DUnet_drop_bn_dice]({{ "/assets/3d-liver-segmentation/3DUnet_bn_drop_pb_dice.png" | absolute_url }})

#### 3D U-net + batch_renorm + dice loss
- `patch_size = (60, 60, 52)`
- `padding = (44, 44, 44)`
- `minibatch_size = 1`
- `learning_rage = 5e-4`
- train only with patches containing liver
![3DUnet_bn_dice]({{ "/assets/3d-liver-segmentation/3DUnet_bn_pb_dice.png" | absolute_url }})

#### 3D U-net + batch_renorm + cce loss
- `patch_size = (60, 60, 52)`
- `padding = (44, 44, 44)`
- `minibatch_size = 1`
- `learning_rage = 5e-4`
![3DUnet_bn_cce]({{ "/assets/3d-liver-segmentation/3DUnet_bn_pb_cce.png" | absolute_url }})

---
#### References
[^1]: Çiçek at al., [*3D U-net: Learning Dense Volumetric Segmentation from Sparse Annotation*](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf). International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer International Publishing, 2016.
