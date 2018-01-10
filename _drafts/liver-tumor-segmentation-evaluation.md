---
layout: post
title: On Evaluation of Liver Tumor Segmentation
categories: segmentation evalation metrics
---

The problem of automatic liver tumor segmentation is a problem which occupies a lot of researches in the medical imaging field. Having a reliable, automatic segmentation tools would allow for better and more accurate diagnosis, therapy planing and therapy response assessment. Currently, the standard way to measure the therapy response is to compare the largest diameters of tumors, which clearly neglects the volume change and texture features. With the recent advancements in the field of machine learning, it is only a matter of time before clinicians would have good automatic segmentation tools at their disposal.

In order to foster research within this particular problem, one needs to define meaningful evaluation criteria the methods are optimized for in order to develop the algorithms in the appropriate direction. By appropriate direction I mean that clinicians would probably prefer an algorithm that finds most of the tumors but segments them poorly than one that finds less with higher segmentation quality. This post aims at summarizing different metrics with their pros and cons.

### Problem
Given **reference** tumor segmentation produced by clinical experts and **test** segmentation output by some algorithm for a group of patients, we would like to measure how good the **test** segmentation is. A patient can have zero or more tumors segmented in the **reference**. The evaluation should consider two points:

- Detection: How good is my method at finding the tumors?
- Segmentation: How well the tumors my method found are segmented?

### Metric Definitions

#### Jaccard index
The Jaccard index[^1] (aka Intersection over Union *IoU*) is a common metric for evaluation of segmentation quality.

$$0 \leq J(A, B) = \frac{|A \cap B|}{|A \cup B|} \leq 1$$

For empty $$A$$ and $$B$$ the $$J(A,B)=0$$. The Jaccard index can be considered a metric, since it satisfies the triangle inequality. Assuming a constant error in terms of missclassified voxels and a variable size of the object to be segmented, the behavior of the Jaccard index as depicted in the following figure can be observed. It is interesting to note, that the Jaccard index favors methods which tend to produce bigger segmentations.
![Jaccard]({{ "/assets/liver-tumor-segmentation-evaluation/jaccard.png" | absolute_url }})

#### Dice index
The Dice index[^2] (aka Dice's coefficient) is a metric defined very similarly to the Jaccard index, but it does not satisfy the triangle inequality.

$$0 \leq DSC(A,B) = \frac{2|A \cap B|}{|A|+|B|} \leq 1$$

Dice and Jaccard indices can be in most situations used interchangeably, since Jaccard can be derived from Dice and vice versa. They exhibit the following relationship:
![JaccardVsDice]({{ "/assets/liver-tumor-segmentation-evaluation/jaccard_vs_dice.png" | absolute_url }})

---
[^1]: [Wikipedia article on Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index)
[^2]: [Wikipedia article on Dice index](https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient)