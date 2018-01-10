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

### Definitions

#### Detection measures
Detection measures are typically functions of real positives $$P$$ (the number of positives in the reference data), real negatives $$N$$, true positives $$TP$$ (the number of correctly found real positives), true negatives $$TN$$, false negatives $$FN$$ (the number of missclassified real negatives) and false positives $$FP$$. Some of the most common measures are Sensitivity, Specificity, Precision and Recall[^3]: 

$$Sensitivity = \frac{TP}{P}$$

$$Specificity = \frac{TN}{N}$$

$$Precision = \frac{TP}{TP + FP}$$

$$Recall = \frac{TP}{P} = \frac{TP}{TP + FN}$$

Precision answers *How many relevant items were selected?*
Recall answers *How many selected items are relevant?*

#### Jaccard index
The Jaccard index[^1] (aka Intersection over Union *IoU*) is a common metric for evaluation of segmentation quality.

$$0 \leq J(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{TP}{TP + FP + FN}\leq 1$$

For empty $$A$$ and $$B$$ the $$J(A,B)=1$$. The Jaccard index can be considered a metric, since it satisfies the triangle inequality. Assuming a constant error in terms of missclassified voxels and a variable size of the object to be segmented, the behavior of the Jaccard index as depicted in the following figure can be observed. It is interesting to note, that the **Jaccard index favours methods which tend to produce bigger segmentations**.
![Jaccard]({{ "/assets/liver-tumor-segmentation-evaluation/jaccard.png" | absolute_url }})

#### Dice index
The Dice index[^2] (aka Dice's coefficient) is a metric defined very similarly to the Jaccard index, but it does not satisfy the triangle inequality.

$$0 \leq DSC(A,B) = \frac{2|A \cap B|}{|A|+|B|} = \frac{2J(A,B)}{1+J(A,B)} = \frac{2TP}{2TP + FP + FN} \leq 1$$

Dice and Jaccard indices can be in most situations used interchangeably, since Jaccard can be derived from Dice and vice versa. They exhibit the following relationship:
![JaccardVsDice]({{ "/assets/liver-tumor-segmentation-evaluation/jaccard_vs_dice.png" | absolute_url }})

#### Free-Response Receiver Operating Characteristic (FROC)
FROC[^4] is an extension to the conventional ROC[^5] which tries to overcome the limitiation of the ROC which is only one decision per case and only two decision alternatives. The FROC allows for analysis of experiments where number of lesions per case may be zero or more and the reader is allowed to take multiple decisions per image. A typical FROC plot would have *Sensitivity* on the y-axis and *FP per image* on the x-axis. It should be noted, that there are currently no established methods to test for significance of differences between two FROC curves as well as no single index summarizing the FROC plot (e.g, ROC has the area under the curve index).

---
[^1]: [Wikipedia article on Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index)
[^2]: [Wikipedia article on Dice index](https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient)
[^3]: [Wikipedia article on Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall)
[^4]: [Extensions to Conventional ROC Methodology: LROC, FROC, and AFROC](https://doi.org/10.1093/jicru/ndn011). Journal of the International Commision on Radiation Units and Measurements, 2008.
[^5]: [Wikipedia article on ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)