---
layout: page
title: Talks
permalink: /talks/
---

### 2019
<div style="display: flex;align-items:stretch;flex-wrap:wrap">
  <div style="display: flex;flex-direction:column;justify-content:center;font-size:60%;margin-bottom:10px">
    <img src="/assets/talks/midl2019/FN_15_relevance.png" style="max-width: 200px;">
  </div>
  <div style="margin-left:10px;flex:1 1 400px;">
    <h4>Relevance Analysis of MRI Sequences for Automatic Liver Tumor Segmentation</h4>
    <p>MIDL 2019 London, England</p>
    <p>
      [<a href="https://openreview.net/pdf?id=B1lRBjMAYN">abstract</a>]
      [<a href="https://github.com/gchlebus/gchlebus.github.io/blob/master/assets/talks/midl2019/midl_poster.pdf">poster</a>]
    </p>
  </div>
</div>

##### Explainability of decisions made by deep neural networks is of high value as it allows for validation and improvement of models. This work proposes an approach to explain semantic segmentation networks by means of layer-wise relevance propagation. As an exemplary application, we investigate which MRI sequences are most relevant for liver tumor segmentation.

### 2018
<div style="display: flex;align-items:stretch;flex-wrap:wrap">
  <div style="display: flex;flex-direction:column;justify-content:center;font-size:60%;margin-bottom:10px">
    <img src="/assets/talks/sirtop-workshop-2018/sirtop-workshop-2018.png" style="max-width: 200px;">
  </div>
  <div style="margin-left:10px;flex:1 1 400px;">
    <h4>Deep Learning Algorithms for Liver and Tumor Segmentation</h4>
    <p><i>Invited talk</i></p>
    <p>SIRTOP Workshop @ Jahreskongress der TGRN e.V. und SRG e.V. in Kooperation mit HGMS e.V., Weimar, Germany</p>
    <p>
      [<a href="https://github.com/gchlebus/gchlebus.github.io/blob/master/assets/talks/sirtop-workshop-2018/sirtop-workshop-2018.pdf">slides</a>]
    </p>
  </div>
</div>

##### Deep learning algorithms have the capability of automating many tedious and time-consuming medical image analysis tasks. In this talk a brief introduction into deep learning will be given as well as an overview of AI applications in the radioembolization context. Moreover, a study regarding automatic deep learning based liver and tumor segmentation in MRI and CT will be presented.

<div style="display: flex;align-items:stretch;flex-wrap:wrap">
  <div style="display: flex;flex-direction:column;justify-content:center;font-size:60%;margin-bottom:10px">
    <img src="/assets/talks/curac2018/liver-net.png" style="max-width: 200px;">
  </div>
  <div style="margin-left:10px;flex:1 1 400px;">
    <h4>Automatic Liver and Tumor Segmentation in Late-Phase MRI Using Fully Convolutional Neural Networks</h4>
    <p>CURAC 2018, Leipzig, Germany</p>
    <p>
      [<a href="https://github.com/gchlebus/gchlebus.github.io/blob/master/assets/talks/curac2018/curac2018.pdf">slides</a>]
    </p>
  </div>
</div>

##### Liver and tumor segmentation plays an important role for many liver interventions. Automation of segmentation steps will bring a substantial improvement to clinical workflows by reducing the planning time and decreasing the inter-observer variability. While liver and tumor segmentation in CT data is a well-studied subject, only few publications are available for MRI data. We present an automatic segmentation approach based on 2D fully convolutional neural networks, which can be applied to both segmentation tasks not only in CT, but also in MR images. For the challenging MRI data, we evaluated two algorithms, one using an axial network and a second one based on three orthogonal models. The latter and better method resulted in a mean Dice index of 0.95 and 0.65 for liver and tumor segmentation, respectively.  The liver segmentation quality of the method matches that of experienced clinical users and is higher compared to other automatic approaches and results in the literature.  The tumor segmentation requires further improvements to make it comparable to the results of human experts.


<div style="display: flex;align-items:stretch;flex-wrap:wrap">
  <div style="display: flex;flex-direction:column;justify-content:center;font-size:60%;margin-bottom:10px">
    <img src="/assets/talks/ComputerVision2.0/neuralnet.png" style="max-width: 200px;">
  </div>
  <div style="margin-left:10px;flex:1 1 400px;">
    <h4>Computer Vision 2.0</h4>
    <p>Science Night @ SBMC 2018, Bremen, Germany</p>
    <p>
      [<a href="https://github.com/gchlebus/gchlebus.github.io/blob/master/assets/talks/ComputerVision2.0/ComputerVision2.0.pdf">slides</a>]
    </p>
  </div>
</div>

##### Deep neural networks have significantly changed the way we design image processing pipelines. Thanks to their ability to accumulate knowledge from big data sets, neural networks are the core of many state-of-the-art computer vision algorithms. In this talk a brief history of computer vision will be given and examplary medical applications will be presented.

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
