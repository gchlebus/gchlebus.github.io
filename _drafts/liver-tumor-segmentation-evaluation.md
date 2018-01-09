---
layout: post
title: On Evaluation of Liver Tumor Segmentation
categories: segmentation evalation metrics
---

The problem of automatic liver tumor segmentation is a problem which occupies a lot of researches in the medical imaging field. Having a reliable, automatic segmentation tools would allow for better and more accurate diagnosis, therapy planing and therapy response assessment. Currently, the standard way to measure the therapy response is to compare the largest diameters of tumors, which clearly neglects the volume change and texture features. With the recent advancements in the field of machine learning, it is only a matter of time before clinicians would have good automatic segmentation tools at their disposal.

In order to foster research within this particular problem, one needs to define meaningful evaluation criteria the methods are optimized for in order to develop the algorithms in the appropriate direction. By appropriate direction I mean that clinicians would probably prefer an algorithm that finds most of the tumors but segments them poorly than one that finds less with higher segmentation quality. This post aims at summarizing different metrics with their pros and cons.

