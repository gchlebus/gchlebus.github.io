---
layout: page
title: Talks
permalink: /talks/
---

### 2017

<div style="display: flex;align-items:stretch;flex-wrap:wrap">
  <div style="display: flex;flex-direction:column;justify-content:center">
    <img src="/assets/talks/LITS_image.png" style="max-width: 200px;">
  </div>
  <div style="margin-left:10px;flex:1 1 400px;">
    <h4>Neural Network Based Automatic Liver and Liver Tumor Segmentation</h4>
    <p>LiTS Workshop @ MICCAI 2017, Quebec, Canada</p>
    <p>
      [<a href="https://github.com/gchlebus/gchlebus.github.io/blob/master/assets/talks/LITS_v3.pdf">slides</a>]
    </p>
  </div>
</div>

##### We present a fully automatic method employing fully convolutional neural networks (FCNNs) and a random forest (RF) classifier to solve the segmentation problems of the 2nd round of the Liver Tumor Segmentation Challenge (LiTS). In order to identify the ROI in which the tumors could be located, a liver segmentation is performed first. For the organ segmentation, an ensemble of FCNNs based on the U-net architecture is trained. Inside of the liver ROI, a 2D FCNN identifies tumor candidates, which are subsequently filtered with a random forest classifier yielding the final tumor segmentation. Our method ranked 3rd according to the segmentation evaluation.

<div style="display: flex;align-items:stretch;flex-wrap:wrap">
  <div style="display: flex;flex-direction:column;justify-content:center">
    <img src="/assets/talks/IGIC_image.png" style="max-width: 200px;">
  </div>
  <div style="margin-left:10px;flex:1 1 400px;">
    <h4>Comparison of Deep Learning and Shape Modeling for Automatic CT-based Liver Segmentation</h4>
    <p>3rd Conference on Image-Guided Interventions, Magdeburg, Germany</p>
    <p>
      [<a href="https://github.com/gchlebus/gchlebus.github.io/blob/master/assets/talks/IGIC_Abstract_v4.pdf">abstract</a>]
      [<a href="https://github.com/gchlebus/gchlebus.github.io/blob/master/assets/talks/IGIC_v1_wo_gif.pdf">slides</a>]
      [<a href="https://github.com/gchlebus/gchlebus.github.io/blob/master/assets/talks/2017_Chlebus_IGIC_Poster_v2.pdf">poster</a>]
    </p>
  </div>
</div>

##### Many liver interventions require an organ segmentation for volumetry and procedure planning. The liver’s varying appearance in CT images makes this organ very time-consuming for manual delineation and challenging for automatic segmentation approaches. Automatic methods are desired, since they allow for a speed-up and reproducibility of the planning process. We investigated two automatic segmentation algorithms based on fully convolutional neural networks (FCN) and statistical shape models (SSM).

<div style="display: flex;align-items:stretch;flex-wrap:wrap">
  <div style="display: flex;flex-direction:column;justify-content:center">
    <img src="/assets/talks/CARS_image.png" style="max-width: 200px;">
  </div>
  <div style="margin-left:10px;flex:1 1 400px;">
    <h4>Comparison of Model Initialization Methods for Liver Segmentation using Statistical Shape Models</h4>
    <p>CARS 2017, Barcelona, Spain</p>
    <p>
      [<a href="https://github.com/gchlebus/gchlebus.github.io/blob/master/assets/talks/CARS_2017_GChlebus_Abstract.pdf">abstract</a>]
    </p>
  </div>
</div>

##### Statistical shape models (SSM) are often employed in automatic segmentation algorithms in order to constrain the domain of plausible results. In a training phase, typical shapes are captured in form of a mean shape and typical modes of variation. At segmentation time, active contour algorithms depend on an appropriate initial placement of a model instance in the image to be segmented. To the best of authors knowledge, the segmentation quality of SSM-based algorithms with respect to the initialization method has received little attention in the literature, although the initialization is indispensable for a fully automatic segmentation. This contribution investigates the influence of different initialization methods for automatic 3D liver segmentation on CT and MR data.