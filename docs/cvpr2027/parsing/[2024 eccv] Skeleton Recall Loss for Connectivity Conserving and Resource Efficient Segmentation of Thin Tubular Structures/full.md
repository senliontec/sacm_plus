# Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures

Yannick Kirchhoff1,2,30\*, Maximilian R. Rokuss1,20\*, Saikat Roy1,20\*, Balint Kovacs1,40, Constantin Ulrich1,40, Tassilo Wald1,2,5, Maximilian Zenk1,4 Philipp Vollmuth1,6,70, Jens Kleesiek8,90, Fabian Isensee1,50, and Klaus Maier-Hein1,10

1 German Cancer Research Center (DKFZ) Heidelberg, Division of Medical Image Computing, Germany

2 Faculty of Mathematics and Computer Science, Heidelberg University, Germany 3 HIDSS4Health - Helmholtz Information and Data Science School for Health, Karlsruhe/Heidelberg, Germany

4 Medical Faculty Heidelberg, Heidelberg University, Heidelberg, Germany

Helmholtz Imaging, German Cancer Research Center, Heidelberg, Germany

6 Division for Computational Radiology Clinical AI (CCIBonn.ai), Clinic for Neuroradiology, University Hospital Bonn, Bonn, Germany 7 Medical Faculty Bonn, University of Bonn, Bonn, Germany

8 Institute for Artificial Intelligence in Medicine (IKIM), University Hospital Essen, Essen, Germany

9 Cancer Research Center Cologne Essen (CCCE), West German Cancer Center Essen, University Hospital Essen, Essen, Germany

10 Pattern Analysis and Learning Group, Department of Radiation Oncology, Heidelberg University Hospital

Abstract. Accurately segmenting thin tubular structures, such as vessels, nerves, roads or concrete cracks, is a crucial task in computer vision. Standard deep learning-based segmentation loss functions, such as Dice or Cross-Entropy, focus on volumetric overlap, often at the expense of preserving structural connectivity or topology. This can lead to segmentation errors that adversely affect downstream tasks, including flow calculation, navigation, and structural inspection. Although current topology-focused losses mark an improvement, they introduce significant computational and memory overheads. This is particularly relevant for 3D data, rendering these losses infeasible for larger volumes as well as increasingly important multi-class segmentation problems. To mitigate this, we propose a novel Skeleton Recall Loss, which effectively addresses these challenges by circumventing intensive GPU-based calculations with inexpensive CPU operations. It demonstrates overall superior performance to current state-of-the-art approaches on five public datasets for topology-preserving segmentation, while substantially reducing computational overheads by more than 90%. In doing so, we introduce the

first multi-class capable loss function for thin structure segmentation, excelling in both efficiency and efficacy for topology-preservation. Our code is available to the community, providing a foundation for further advancements, at: https://github.com/MIC-DKFZ/Skeleton-Recall.

Keywords: Segmentation · Topology · Tubular Structures · Loss Function

## 1 Introduction

The precise segmentation of thin tubular structures is a critical task across diverse domains in engineering and medical applications (Fig. 1). Topological correctness is fundamental for facilitating downstream tasks such as analyzing blood flow dynamics, delineating neuronal boundaries in Electron Microscopy imagery evaluating risk factors for vascular pathologies, aiding in surgical planning, and optimizing route planning [1, 5, 6, 21, 32, 40]. Classical approaches for the automated segmentation of thin curvilinear structures have encompassed methods including image transforms [25, 34], mathematical morphologies [29, 41], filtering [10,13,15], differential operators [33], among others [2,3,8,16,19]. Deep learning based techniques have played an increasing role in recent years with standard segmentation networks such as UNet [27] being popular. Standard overlap-based losses (eg. dice-similarity coefficient [43) enable such networks to segment large structures while often struggling with small elongated ones [18] as shown in Fig. 2.

Multiple methods have been introduced in recent years to address the challenges of segmenting thin curvilinear structures but are often domain-specific or require the use of specialized networks [4, 9, 17, 22–24, 39]. Recently, centerlineDice [31] (clDice) was introduced encompassing both a loss function and a metric for measuring connectivity in segmentation of thin structures. Effectively, it incorporates the skeleton of a segmentation into the dice calculation. While the clDice metric uses the exact skeleton, the clDice loss works with a differentiable approximation of the skeleton. It is used to enable architectureindependent, topology-aware segmentation of thin tubular structures and is considered as state-of-the art. However, despite its advantages, it introduces a large computational overhead. Furthermore, the differentiable Soft Skeleton used in the loss calculation can often be jagged as depicted in Fig. 3, leading to inaccuracies in segmentation. While a follow-up approach [20] attempted to address this limitation by introducing a topological-correct differentiable skeleton, it did so while being even more computationally expensive. This limitation becomes particularly pronounced when dealing with large volumes or multi-class segmentation problems common to 3D medical image segmentation, rendering training on multi-class 3D datasets challenging to infeasible even on modern hardware.

In response to these challenges, we propose Skeleton Recall Loss, a novel loss function tailored to address the intricacies associated with thin structures in segmentation tasks. Skeleton Recall Loss demonstrates the following strengths:

![](images/39d03169eb65d10df59915a9c78c56d8e7f82fbd17c066c59ad7fb3d97e4a751.jpg)  
(a) Roads

![](images/56cf362335ae217fad856b163aaaf63e73675ec7906d2a8fc1df850b07f56c06.jpg)  
(b) DRIVE

![](images/d63af3b4bb254161c940cf528e3ea98ff843f2f90b91ac5fa9e419a125471ba5.jpg)  
(c) Cracks

![](images/bb0844c6ffce2295edc72fca08fa05a13399e7dba12d21b386c940902c389f0a.jpg)  
(d) Toothfairy

![](images/1454b3c7cc71e490c4a57ac8c59221792868312ea97b8aa0daba50aac733eac1.jpg)  
(e) TopCoW  
Fig. 1: Diversity of thin structures. Segmentation of thin structures is a challenging task in engineering and medical imaging. This is highlighted in 5 diverse datasets used in this work to incorporate the segmentation of: a) Roads in satellite imagery, b) Retinal vessels, c) Cracks in concrete structures, d) Inferior alveolar canal in facial CTs, and e) Circle of Willis arterial vessel components.

1. Minimal training time: The Tubed Skeleton required by Skeleton Recall Loss can be computed with simple CPU-based operations using common image processing frameworks (e.g. scikit-image [38]) as part of data-loading or even be precomputed. It is then used in a simple additional soft recall loss with the prediction, thus requiring very little additional training time.

2. Minimal training memory: The utilization of Skeleton Recall Loss entails minimal GPU Memory overhead. Unlike approaches reliant on a differentiable skeleton in prediction or ground truth during training, Skeleton Recall Loss sidesteps the computationally taxing GPU-based skeletonization process, thus necessitating only a marginal increase in GPU training memory.

3. Domain and architecture agnostic: Skeleton Recall Loss exhibits inherent plug-and-play characteristics, seamlessly integrating into a wide array of 2D and 3D segmentation tasks without imposing architectural constraints. It operates without the need for specialized networks or modifications to underlying segmentation architectures.

4. Multi-class compatibility: Skeleton Recall Loss integrates seamlessly with multi-class labels, while competing methods like clDice Loss, often face near insurmountable computational challenges on such problems.

Skeleton Recall Loss yields overall superior results to a baseline network without topological losses, as well as against clDice Loss as a state-of-the-art topological loss. We demonstrate this effectiveness on extensive multi-domain evaluation on 5 publicly available datasets. Notably, our loss function inherently feasibly supports multi-class segmentation problems and thus can be considered a new state-of-the-art for dilineating thin curvilinear structures in natural as well as medical images.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Dice Loss</td><td rowspan=1 colspan=1>cl-Dice Loss</td><td rowspan=1 colspan=1>Skeleton Recall Loss</td></tr><tr><td rowspan=1 colspan=1>Connectivity EnhancingResource EfficientMulti-Class Capable</td><td rowspan=1 colspan=1>XV0</td><td rowspan=1 colspan=1>VXX</td><td rowspan=1 colspan=1><img src="images/04f393a6d0ab8552519503f9931c05af3ae88db97af123aa8b368f043386327c.jpg"/></td></tr><tr><td rowspan=1 colspan=1><img src="images/ff4c95ff0a8165f0985e2ed8e4c9aa33e11f95e738a2d44c27f6f950efc452eb.jpg"/></td><td rowspan=1 colspan=1><img src="images/3b9f9ea35c86f874a7976186b90c1f420b4644adf9fe509b8830ddec177fc027.jpg"/></td><td rowspan=1 colspan=2><img src="images/cba2d56845ad4eee5c7cc820231838efb1bac9ad7a14a29c5eb27540e9af0221.jpg"/>                      <img src="images/cd23394348b5bffb146397cdd7f50364dfa2def9044c56c11e087969b8f1ceff.jpg"/></td></tr></table>

Fig.2: Comparison of state-of-the-art loss functions on the task of thin structure segmentation. Top: Our Skeleton Recall Loss efficiently addresses connectivity conservation, unlike standard dice loss, without the overhead of clDice Loss, making it ideal for multi-class problems as well. Bottom: Qualitative results on the TopCoW [40] dataset. Due to computational cost clDice Loss can not be used for multi-class segmentation.

## 2 Related Work

Deep learning-based approaches for segmentation of thin curvilinear structures often involve specialized networks. In [22], a joint network was trained with a shared encoder to use 2 decoders to simultaneously segment as well as score a tubular path. Similarly, in [4], a joint network was proposed to simultaneously learn features as well as global topology. Also, in [17], a pair of sequential UNets was used, the first of which performed a coarse prediction while the other detected missing or false splits in the structure. A similar end-to-end approach in [24] used a channel and spatial attention module, which was incorporated within the bottleneck of an encoder-decoder network for segmenting thin structures in medical images. In another work [26], an oriented derivative of stick (ODoS) filter output was refined using a succession of UNets to obtain effective segmentation of curvilinear objects. However, alternative approaches utilize specialized loss functions to assist the network in preserving topology in segmentation outputs. In [7], topological priors were incorporated into network training by the usage of a persistent-homology based differentiable topological prior as a loss function. Another work [11], attempted to preserve topological information by enforcing that the prediction and ground truth have the same Betti number via a novel loss function. Recently, the use of a differentiable skeleton has emerged as the predominant method for topology-aware segmentation of thin structures following the introduction of the centerlineDice (clDice) loss function [31], outperforming persistent-homology based approaches. This method is complemented by the introduction of the clDice metric, a well-established measure of connectedness. The differentiable skeleton proposed by [31] was improved in [37] where a Soft-Persistent Skeleton was proposed for coronal artery tracking. Alternatively, in [28], the differentiable skeleton was predicted by a secondary network in addition to the primary segmentation output. Most recently, a topologically correct differentiable skeletonization algorithm was introduced in [20], overcoming previous approximation of skeletons, while still requiring massive computational resources to do so.

![](images/c2931c3153cb2997a1b896fe1ac0de9dab2ab56c74e133e6c2753dce94396753.jpg)  
(a) MRA Image

![](images/2bbb6488ae830bcb68b74aebbe2e704edf9e95d2fb3a44e9f49b4bf8e1f9de5a.jpg)  
(b) GT segmentation

![](images/1d6266f8d354ff2ee3d67b8c744dc90792c03afe89fac0131b47f7a959f75f19.jpg)  
(c) Soft Skeleton [31]

![](images/b9ce41c4e46a5dd0428013e3f165ac9d7769924739a92167ceeed62be13e5035.jpg)  
(d) Tubed Skeleton  
Fig. 3: The challenges of Differentiable Skeletons. Visual comparison of (c) the soft skeleton used for the calculation of the clDice Loss [31] and (d) the proposed tubed skeleton used for Skeleton Recall Loss for (a) an image and the corresponding (b) ground truth segmentation, originating from the TopCoW dataset [40].

## 3 Methodology

## 3.1 The challenges of Differentiable Skeletons

The usage of a differentiable skeleton based loss [20, 28, 31, 37] in training a deep neural network to segment thin tubular structures is an intuitive approach to preserve connectivity. However, it is fraught with challenges which can be multi-faceted in nature. One of the most easily demonstrable issues is shown in Fig. 3. As mentioned earlier, the so-called Soft skeletonization of clDice Loss can lead to perforated and jagged skeletons, especially in 3D, which results in inaccuracies for the clDice Loss calculation. This is in addition to the enormous GPU memory and training time overheads that are a natural part of a GPUbased differentiable skeletonization process on both the ground truth as well as the network prediction. This overhead is demonstrated in Fig. 7, and can render effective training almost infeasible in multi-class datasets with large 3D input volumes such as TopCoW [40] (which is used in this work) without access to significant computational resources. While follow-up work in [20] allowed for a relative improvement in topologically accurate differentiable skeletonization, it further aggravated the issues with excessive resource utilization.

## 3.2 Skeleton Recall Loss: Connectivity conservation on thin structures without differentiable skeletons

Skeleton Recall Loss is a loss function designed to preserve connectivity in thin tubular structures without incurring massive computational overheads. It is universally applicable, regardless of whether the inputs are 2D or 3D. It does so by avoiding the GPU-based soft-skeletonization on the prediction and ground truth. Instead, a tubed skeletonization is performed on the ground truth, followed by a soft recall loss against the predicted segmentation output. This is illustrated in Fig. 4 and further discussed in the following sections.

![](images/eab5d977a1d8d690beb38cba75312cb3717bb7287278fc8cc9e57ea2f13afe7f.jpg)  
Fig. 4: Overview of our method in comparison to differentiable skeleton based approaches. Initially, a segmentation network (green) predicts a segmentation mask. Our proposed Skeleton Recall Loss (blue) calculates the soft recall of the prediction on the precomputed Tubed Skeleton of the ground truth. In doing so, we mitigate the massive overheads introduced by differentiable skeleton based methods (red).

Tubed Skeletonization The usage of a skeleton for the preservation of connectivity is an effective method, but it does not need to be differentiable. In this work, we extract a tubed skeleton from the ground truth as demonstrated in Algorithm 1. Initially, we binarize the ground truth segmentation mask and compute its skeleton using methods outlined in [42] for 2D and [14] for 3D inputs. Subsequently, we dilate the skeleton with a diamond kernel of radius 2 to make it tubular, thereby enlarging the effective area for loss computation around the otherwise thin, single-pixel-wide skeleton. This enhances the stability of the loss by incorporating signals from a greater number of pixels, particularly those in close proximity to the skeleton, which are vital for connectivity. Lastly, for multi-class problems, we multiply the tubed skeleton with the ground truth mask, effectively assigning parts of the skeleton to their respective classes. All of these operations are computationally inexpensive and can be carried out on the CPU during data loading or pre-computed using libraries such as scikit-image [38].

Algorithm 1 Tubed Skeletonization   
Require: Y are K-classed hard targets where $Y _ { i , j ( , k ) } \in [ 0 , K ]$   
1: $Y _ { \mathrm { b i n } }  Y > 0$ % Binarize to foreground and background labels   
2: Yskel ← skeletonize $( Y _ { \mathrm { b i n } } )$ % Extract binarized skeleton   
3: $Y _ { \mathrm { s k e l } }  \mathbf { d i l a t e } ( Y _ { \mathrm { s k e l } } )$ % Dilate to create tubed skeleton   
4: $Y _ { \mathrm { m c - s k e l } }  Y _ { \mathrm { s k e l } } \times Y$ % De-binarize to create multi-class tubed skeleton   
5: return $Y _ { \mathrm { m c - s k e l } }$

Soft Recall on Tubed Skeleton Following the extraction of our tubed skeleton, we incentivize the network to include as much of this skeleton as possible as part of the prediction. This is performed simply by using a soft recall loss $\mathcal { L } _ { S k e l R e c a l l }$ (Eq. (1)), in addition to any existing generic loss $\mathcal { L } _ { g e n e r i c }$ used by the network (for example, Dice Loss, Cross Entropy Loss, etc.).

$$
\mathcal { L } _ { S k e l R e c a l l } = - \frac { 1 } { | C | } \sum _ { c \in C } \frac { \sum _ { i } Y _ { \mathrm { s k e l } , i , c } \cdot \hat { Y } _ { i , c } } { \sum _ { i } Y _ { \mathrm { s k e l } , i , c } }\tag{1}
$$

This vastly improves the connectivity of thin curvilinear structures predicted by a network trained using this loss (Sec. 5.1). Additionally, Skeleton Recall Loss is computationally inexpensive in comparison to the use of a differentiable skeletonization, requiring only fractionally more GPU memory and additional time during training (Sec. 5.3). This facilitates training multi-class segmentation problems as opposed to current differentiable skeleton methods which incur infeasible overheads (Sec. 5.3).

## 4 Experimental Setup

## 4.1 Datasets

We employ five public datasets featuring thin structures for validating the proposed Skeleton Recall Loss. The datasets span natural as well as medical images, covering a range of segmentation challenges, including both binary and multiclass segmentation problems in 2D as well as 3D contexts. An overview of the datasets can be found in Tab. 1. Among the three 2D datasets used in this study, the Digital Retinal Images for Vessel Extraction (DRIVE) dataset [32] was employed, focusing on retinal vessel segmentation. Additionally, structural inspection images designed for concrete crack segmentation (Cracks) [36] and aerial images of Massachusetts for road segmentation (Roads) [21| were included, highlighting the diversity of thin structures in natural and constructed environments. In the 3D domain, we incorporated two cutting-edge medical image segmentation challenge datasets. One of them was ToothFairy¹1, which was a segmentation challenge on 3D Cone-Beam CTs [5,6] featuring the inferior alveolar canal as the target structure. Additionally, the TopCoW12 dataset for topology-aware 3D segmentation of vessels in the Circle of Willis for CTA and MRA data [40] was utilized, encompassing binary as well as multi-class segmentation on 13 different subtypes of vessels. This diverse set of datasets enables a comprehensive evaluation of the proposed Skeleton Recall Loss, demonstrating generalizability of the method to a wide range of thin structure segmentation challenges in both 2D and 3D contexts.

Table 1: Details of the datasets used for training and evaluation. Our datasets show wide coverage over a number of thin structure segmentation tasks in natural and medical images. The TopCoW dataset is used both in binary and multi-class settings, in line with the original challenge.
<table><tr><td>Dataset</td><td>Dims</td><td>Type</td><td> $\#$  Samples</td><td>Target Structure</td></tr><tr><td>Roads [21]</td><td>2D</td><td>binary</td><td>804</td><td>Roads on aerial images</td></tr><tr><td>DRIVE [32]</td><td>2D</td><td>binary</td><td>20</td><td>Blood vessels on retina images</td></tr><tr><td>Cracks [36]</td><td>2D</td><td>binary</td><td>5388</td><td>Cracks on concrete structure images</td></tr><tr><td>ToothFairy [5, 6]</td><td>3D</td><td>binary</td><td>138</td><td>Inferior Alveolar Canal</td></tr><tr><td>TopCoW [40]</td><td>3D</td><td>multi-class</td><td>200</td><td>Circle of Willis vessels in the brain</td></tr></table>

## 4.2 Evaluation Metrics

We use multiple metrics including overlap, connectivity and topological measures for thorough evaluation of our proposed loss function. An interesting dichotomy of clDice [31] is that while it makes for an inefficient loss function for training deep neural networks, it is an effective metric for measuring connectivity. Therefore, following existing guidelines [18] for semantic segmentation of tubular structures, we use clDice as a metric in conjunction with Dice similarity coefficient, as our connectivity- and overlap-based measures. Similar to previous work, we also report on topology-based metrics, namely, absolute Betti Number Errors of $0 ^ { \mathrm { t h } }$ and $1 ^ { \mathrm { s t } }$ Betti Numbers, $\beta _ { 0 }$ and $\beta _ { 1 }$ . However, in contrast to other work [11, 31], we calculate the Betti Errors on whole volumes instead of small, randomly extracted patches. Our evaluation strategy is more intuitive in nature and offers better interpretability of the measure, which is especially relevant in medical segmentation tasks.

## 4.3 Baseline Loss Functions

We benchmark our proposed loss function against state-of-the-art loss functions targeting thin structure segmentation on the five datasets detailed in Sec. 4.1. Specifically, we compare against: 1) clDice Loss [31], the leading method in the field that utilizes approximate differentiable skeletons. 2) Additionally, we also compare against a modification of clDice Loss, where we replace the differentiable skeletonization of the original publication by a follow-up of this work [20]. This new method, called Topo-clDice Loss in our evaluations, produces topologically-accurate differentiable skeletons at the cost of even higher computational requirements. We note that loss functions based on persistent homologies [11, 22] are excluded from our evaluation which, while related, were surpassed by the clDice Loss [31].

## 4.4 Training

We implement the baseline loss functions (Sec. 4.3), as well as our proposed Skeleton Recall Loss in a powerful medical image segmentation network (nnUNet [12]) and a state-of-the-art natural image segmentation network (HRNet [35]), pretrained on ImageNet [30]. We use the examined loss functions for connectivity conservation $( \mathcal { L } _ { c o n n e c t i v i t y } )$ in addition to the underlying generic loss $( \mathcal { L } _ { g e n e r i c } )$ of our training framework – a combination of Cross-Entropy and Soft Dice Loss. The connectivity loss is weighted by an additional parameter w as shown in Eq. (2).

$$
\mathcal { L } = \mathcal { L } _ { g e n e r i c } + w \cdot \mathcal { L } _ { c o n n e c t i v i t y }\tag{2}
$$

Our experiments are restricted to two weight configurations $w \in \{ 0 . 1 , 1 . 0 \}$ in order to curb the influence of extensive hyperparameter tuning. We show a more detailed analysis of the effect of the weight parameter in the Appendix. Additionally, the full set of hyperparameters, optimizers and configurations of nnUNet and HRNet used for training on the different datasets are also provided in the Appendix.

## 5 Results and Discussion

## 5.1 Skeleton Recall Loss enables state-of-the-art segmentation of thin structures

The obtained results in Tab. 2 clearly show that our proposed Skeleton Recall Loss consistently surpasses previous thin structure segmentation losses on almost all datasets. For concrete crack segmentation [36], the results indicate better Dice and clDice performance at the cost of slightly worse Betti numbers than clDice Loss. However, Skeleton Recall Loss demonstrates the best clDice and Betti numbers for retinal vessel segmentation [32], yielding a Dice score just marginally behind clDice Loss. Notably, for the three datasets with an independent testset available, specifically Roads [21] and both of the 3D datasets, ToothFairy [5, 6] and TopCoW [40], we observe superior performance of our proposed Skeleton Recall Loss. This is further demonstrated by the qualitative results given in Fig. 5. Skeleton Recall Loss is also seen to be better than baselines, on both binary as well as multi-class settings of TopCoW as elaborated in following sections. We obtain this state-of-the-art performance while being architecture agnostic (Sec. 5.2) as well as overwhelmingly resource efficient (Sec. 5.3).

Table 2: State-of-the-art segmentation of thin structures. Quantitative results obtained by incorporating our proposed Skeleton Recall Loss as well as existing thin structure segmentation losses into the loss function of a generic nnUNet backbone. Results are reported on the testset, except for DRIVE and Cracks datasets, where we report 5-fold cross validation results due to unavailability of an independent testset.
<table><tr><td>Dataset</td><td>Loss configuration</td><td>Dice ↑ clDice ↑</td><td>βo error ↓ β1 error ↓</td><td></td></tr><tr><td rowspan="4">Roads [21]</td><td>Default nnUNet</td><td>78.99 88.79</td><td>5.769</td><td>84.62</td></tr><tr><td>+ clDice Loss</td><td>79.15 89.00</td><td>6.539</td><td>82.00</td></tr><tr><td>+ Topo-clDice Loss</td><td>78.94 88.62</td><td>11.00</td><td>85.92</td></tr><tr><td>+ Skeleton Recall Loss (Ours)</td><td>79.25 89.06</td><td>4.846</td><td>83.69</td></tr><tr><td rowspan="4">DRIVE [32]</td><td>Default nnUNet</td><td>80.87 80.26</td><td>57.00</td><td>22.80</td></tr><tr><td>+ clDice Loss</td><td>81.05 80.68</td><td>44.50</td><td>23.35</td></tr><tr><td>+ Topo-clDice Loss</td><td>80.80 80.19</td><td>46.35</td><td>23.70</td></tr><tr><td>+ Skeleton Recall Loss (Ours)</td><td>80.99 80.83</td><td>38.75</td><td>21.50</td></tr><tr><td rowspan="4">Cracks [36]</td><td>Default nnUNet</td><td>94.59 95.76</td><td>0.147</td><td>0.0033</td></tr><tr><td>+ clDice Loss</td><td>94.80 95.96</td><td>0.142</td><td>0.0033</td></tr><tr><td>+ Topo-clDice Loss</td><td>94.83 96.00</td><td>0.159</td><td>0.0035</td></tr><tr><td>+ Skeleton Recall Loss (Ours)</td><td>94.88 96.04</td><td>0.148</td><td>0.0035</td></tr><tr><td rowspan="3">ToothFairy [5,6]</td><td>Default nnUNet</td><td>71.80 89.16</td><td>0.900</td><td>0.0200</td></tr><tr><td>+ clDice Loss</td><td>72.36 89.67</td><td>0.620</td><td>0.0200</td></tr><tr><td>+ Skeleton Recall Loss (Ours)</td><td>74.42 92.05</td><td>0.540</td><td>0.0200</td></tr><tr><td rowspan="3">TopCoW binary [40]</td><td>Default nnUNet</td><td>93.55 98.25</td><td>0.743</td><td>1.800</td></tr><tr><td>+ clDice Loss</td><td>93.64 98.35</td><td>0.514</td><td>1.986</td></tr><tr><td>+ Skeleton Recall Loss (Ours)</td><td>93.72 98.48</td><td>0.500</td><td>1.586</td></tr><tr><td rowspan="3">TopCoW multi-class</td><td>Default nnUNet</td><td>85.36 93.68</td><td>0.137</td><td>0.0571</td></tr><tr><td>+ clDice Loss</td><td></td><td>– Out Of Memory −</td><td></td></tr><tr><td>+ Skeleton Recall Loss (Ours)</td><td>86.59 94.35</td><td>0.151</td><td>0.056</td></tr></table>

## 5.2 Skeleton Recall Loss is architecture agnostic

While Skeleton Recall Loss demonstrates state-of-the-art performance using nnUNet as a backbone framework in Tab. 2, it is not restricted to specialized architectures. We highlight this in Tab. 3 where HRNet [35], a state-of-the-art 2D architecture for natural image segmentation, is used as the backbone. This leads to similar benefits on connectivity conservation using Skeleton Recall Loss during training on our 2D datasets. Skeleton Recall Loss is seen to exceed the connectivity conserving performance (as demonstrated by the clDice metric) on 2 out of 3 datasets, while being comparable on the remaining one. Our overall superiority over all metrics demonstrates that Skeleton Recall Loss is architecture agnostic and can be used as a loss in training arbitrary deep architectures for connectivity-conserving segmentation of thin structures.

## 5.3 Connectivity conservation with minimal overheads

Efficient binary segmentation of thin structures A plurality of tasks in the segmentation of thin curvilinear structures have historically been binary in nature. As competing state-of-the-art differentiable skeleton methods were developed for the binary scenario, we consider this to be where such methods should also be most competitive. However, Skeleton Recall Loss does not only provide state-of-the-art connectivity-conserving thin structure segmentation performance, as seen in Sec. 5.1 and Sec. 5.2, but it can do so while using only fractional GPU memory and training time compared to existing methods, as shown in Fig. 6. Differentiable skeleton based methods require a GPU-based skeleton computation [31] or prediction [28]. For our differentiable skeleton baseline clDice Loss, this leads to approximately 88% additional training time and 52% more VRAM consumption compared to the plain nnUNet backbone when averaged across our 5 datasets (excluding multi-class TopCoW). Remarkably, our method Skeleton Recall Loss does the same at only an additional 8% training time and 2% higher VRAM consumption. This illustrates that Skeleton Recall

Image  
Ground Truth  
nnUNet  
+ clDice Loss  
+ Ours  
![](images/155bb68f70ad7d18721b4547bc6948c624df6e19cb20c1418c9ab9f84e1ab82c.jpg)  
Fig. 5: Connectivity conservation in qualitative results on 5 datasets. nnUNet with conventional segmentation losses performs well in adequately delineating general structures, particularly thicker ones. However, challenges arise in accurately capturing thin structures and maintaining connectivity within the segmentation. This is demonstrated on examples from (top to bottom) Roads, DRIVE, Cracks, Toothfairy and TopCoW datasets. Augmenting the model with clDice Loss yields some improvement but falls short in addressing connectivity issues. In contrast, our proposed Skeleton Recall Loss demonstrates enhanced preservation of topology and improved connectivity in segmentation outputs.

Table 3: Skeleton Recall Loss is architecture agnostic. Quantitative results using HRNet, a state-of-the-art 2D network, on all examined 2D datasets. Skeleton Recall Loss demonstrates accurate segmentation including effective connectivity conservation, without explicit reliance on a particular deep neural network architecture.
<table><tr><td colspan="2">Dataset Loss configuration</td><td>Dice ↑ clDice ↑</td><td> $\beta _ { \mathbf { 0 } }$ </td><td>error ↓ β1 error ↓</td></tr><tr><td rowspan="3">Roads</td><td>HRNet</td><td>77.91 87.63</td><td>18.08</td><td>83.15</td></tr><tr><td>+ clDice Loss</td><td>78.05 88.21</td><td>8.846</td><td>82.31</td></tr><tr><td>+ Skeleton Recall Loss (Ours)</td><td>78.25 88.14</td><td>15.38</td><td>81.31</td></tr><tr><td rowspan="3">DRIVE</td><td>HRNet</td><td>80.34 77.83</td><td>149.3</td><td>51.25</td></tr><tr><td>+ clDice Loss</td><td>80.63 81.65</td><td>65.75</td><td>46.25</td></tr><tr><td>+ Skeleton Recall Loss (Ours)</td><td>76.08 84.20</td><td>27.25</td><td>38.50</td></tr><tr><td rowspan="3">Cracks</td><td>HRNet</td><td>94.09 95.39</td><td>0.1985</td><td>0.0037</td></tr><tr><td>+ clDice Loss</td><td>94.67 96.01</td><td>0.1558</td><td>0.0037</td></tr><tr><td>+ Skeleton Recall Loss (Ours)</td><td>95.03 96.26</td><td>0.1596</td><td>0.0019</td></tr></table>

![](images/0fe931997f883d9b666e13d0b95298b20cd3c9a61c988ec03d7626637d07f668.jpg)

![](images/5ab5c625bd8be83191bf0e5c1a4aa77cdb9f64c387d3609bd91f864e3b7065eb.jpg)  
Fig. 6: Efficient Resource Utilization for Binary Segmentation. The figures depict the additional training time per epoch and memory requirements caused by employing Skeleton Recall Loss and clDice Loss compared to the standard network training (nnUnet, dashed lines) across all assessed datasets. While Skeleton Recall Loss shows minimal increase in VRAM usage and negligible changes in epoch duration, clDice Loss introduces notable overhead in both time and memory. For example, clDice Loss more than doubles the epoch duration for DRIVE or almost doubles VRAM usage for Toothfairy.

Loss categorically outperforms traditional differentiable skeleton-based methods on binary settings, which they were developed for, in terms of resource efficiency.

Enabling multi-class segmentation of thin structures Binary segmentation has historically sufficed for many image analysis tasks across various domains. However, as the demand for finer-grained analysis grows, transitioning to multi-class segmentation becomes increasingly vital. This shift is especially pertinent in medical contexts due to the prevalence of thin structures where binary segmentation may not adequately capture the complexity of anatomical features. For instance, the recent TopCoW challenge [40] revealed that binary segmentation of brain vessels can be deemed as sufficiently solved, approaching inter-rater agreement in the Dice score. However, differentiating between the different vessels still remains a challenging task. Our Skeleton Recall Loss demonstrates powerful multi-class segmentation capabilities in addition to standard binary settings. Tab. 2 showcases the results of multi-class segmentation on 13 different brain vessel classes of the TopCoW dataset using both standard nnUNet and our proposed Loss. The results demonstrate that while nnUNet exhibits slightly better $\beta _ { 0 }$ error, our Skeleton Recall Loss significantly improves Dice and clDice scores. Moreover, it performs on par in terms of $\beta _ { 1 }$ error, ultimately yielding a superior overall result.

Fig. 7 shows the multi-class resource utilization with respect to the number of classes of our proposed Loss in comparison to clDice Loss. We demonstrate significant training time and memory savings with near-constant additional overhead despite the increasing number of classes. In contrast, the plots underscore the approximately linear growth in memory consumption and training time associated with clDice Loss. We note that the inefficiency of clDice Loss rendered it infeasible on all 13 classes as it exceeded the memory capacity of an A100 40GB GPU. In summary, Skeleton Recall Loss can be employed for an arbitrary number of classes with minimal computational cost.

## 6 Conclusion

This paper proposes a novel loss function, Skeleton Recall Loss, designed for connectivity preserving semantic segmentation. It is domain and architecture agnostic and, unlike existing methods, requires minimal additional training time and memory. Through extensive evaluation on five publicly available datasets, we demonstrate that Skeleton Recall Loss shows overall superior performance on existing state-of-the-art topology-aware loss functions. Moreover, it stands as the first loss function designed for computationally manageable thin structure segmentation within the increasingly significant but hitherto unaddressed multi-class context. In essence, Skeleton Recall Loss represents a significant advancement in the field of thin structure segmentation, offering both efficiency and efficacy. The public availability of our code further facilitates its adoption and serves as a foundation for future advancements in this critical area of study.

![](images/a5c2c28d6954b18196b8f976de0a7e9ad461ede7a1ea8ad123f5a933d426d36f.jpg)

![](images/1f0c8ec1dde93ee0ba90ef0ce642600e0b30183ac1b76da723b83fbcf8bea40f.jpg)

![](images/8ba6b1778cf6939e9543a9a96cb0743e1dd2daa22795e6022444c3150092e824.jpg)

![](images/317222cbc461126703e645a9715000feaecdce531cbb68f58b588de53684d56c.jpg)  
Fig. 7: Resource Utilization for Multi-class Segmentation. Skeleton Recall Loss requires minimal additional GPU memory and training time for an increasing number of classes, as it avoids a differentiable skeleton computation. Competing methods like clDice Loss, on the other hand, incur enormous overheads which make multi-class training on datasets such as TopCoW [40] infeasible. Analysis was performed for two different batch sizes (BS) on a single A100 40GB GPU, averaged over 5 training epochs, for ease of comparability.

## Acknowledgement

The present contribution is supported by the Helmholtz Association under the joint research school "HIDSS4Health – Helmholtz Information and Data Science School for Health". This work was partly funded by Helmholtz Imaging (HI), a platform of the Helmholtz Incubator on Information and Data Science. PV is funded through an Else Kröner Clinician Scientist Endowed Professorship by the Else Kröner Fresenius Foundation (reference number: 2022 EKCS.17).

## References

1. Arganda-Carreras, I., Turaga, S.C., Berger, D.R., Cireşan, D., Giusti, A., Gambardella, L.M., Schmidhuber, J., Laptev, D., Dwivedi, S., Buhmann, J.M., et al.: Crowdsourcing the creation of image segmentation algorithms for connectomics. Frontiers in neuroanatomy 9, 142 (2015)

2. Bibiloni, P., González-Hidalgo, M., Massanet, S.: A survey on curvilinear object segmentation in multiple applications. Pattern Recognition 60, 949–970 (2016)

3. Chambon, S., Moliard, J.M.: Automatic road pavement assessment with image processing: Review and comparison. International Journal of Geophysics 2011 (2011)

4. Cheng, M., Zhao, K., Guo, X., Xu, Y., Guo, J.: Joint topology-preserving and feature-refinement network for curvilinear structure segmentation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 7147–7156 (2021)

5. Cipriano, M., Allegretti, S., Bolelli, F., Di Bartolomeo, M., Pollastri, F., Pellacani, A., Minafra, P., Anesi, A., Grana, C.: Deep segmentation of the mandibular canal: a new 3d annotated dataset of cbct volumes. IEEE Access 10, 11500–11510 (2022)

6. Cipriano, M., Allegretti, S., Bolelli, F., Pollastri, F., Grana, C.: Improving segmentation of the inferior alveolar nerve through deep label propagation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21137–21146 (2022)

7. Clough, J.R., Byrne, N., Oksuz, I., Zimmer, V.A., Schnabel, J.A., King, A.P.: A topological loss function for deep-learning based image segmentation using persistent homology. IEEE transactions on pattern analysis and machine intelligence 44(12), 8766–8778 (2020)

8. Fraz, M.M., Remagnino, P., Hoppe, A., Uyyanonvara, B., Rudnicka, A.R., Owen, C.G., Barman, S.A.: Blood vessel segmentation methodologies in retinal images-a survey. Computer methods and programs in biomedicine 108(1), 407–433 (2012)

9. He, Y., Sun, H., Yi, Y., Chen, W., Kong, J., Zheng, C.: Curv-net: curvilinear structure segmentation network based on selective kernel and multi-bi-convlstm. Medical Physics 49(5), 3144–3158 (2022)

10. Hoover, A., Kouznetsova, V., Goldbaum, M.: Locating blood vessels in retinal images by piecewise threshold probing of a matched filter response. IEEE Transactions on Medical imaging 19(3), 203–210 (2000)

11. Hu, X., Li, F., Samaras, D., Chen, C.: Topology-preserving deep image segmentation. Advances in neural information processing systems 32 (2019)

12. Isensee, F., Jaeger, P.F., Kohl, S.A., Petersen, J., Maier-Hein, K.H.: nnu-net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods 18(2), 203–211 (2021)

13. Koller, T.M., Gerig, G., Szekely, G., Dettwiler, D.: Multiscale detection of curvilinear structures in 2-d and 3-d image data. In: Proceedings of IEEE International Conference on Computer Vision. pp. 864–869. IEEE (1995)

14. Lee, T.C., Kashyap, R.L., Chu, C.N.: Building skeleton models via 3-d medial surface axis thinning algorithms. CVGIP: Graphical Models and Image Processing 56(6), 462–478 (1994)

15. Lemaitre, C., Perdoch, M., Rahmoune, A., Matas, J., Mitéran, J.: Detection and matching of curvilinear structures. Pattern recognition 44(7), 1514–1527 (2011)

16. Lesage, D., Angelini, E.D., Bloch, I., Funka-Lea, G.: A review of 3d vessel lumen segmentation techniques: Models, features and extraction schemes. Medical image analysis 13(6), 819–845 (2009)

17. Lin, M., Zepf, K., Christensen, A.N., Bashir, Z., Svendsen, M.B.S., Tolsgaard, M., Feragen, A.: Dtu-net: Learning topological similarity for curvilinear structure segmentation. In: International Conference on Information Processing in Medical Imaging. pp. 654–666. Springer (2023)

18. Maier-Hein, L., Reinke, A., Godau, P., Tizabi, M.D., Buettner, F., Christodoulou, E., Glocker, B., Isensee, F., Kleesiek, J., Kozubek, M., et al.: Metrics reloaded: recommendations for image analysis validation. Nature methods pp. 1-18 (2024)

19. Mena, J.B.: State of the art on automatic road extraction for gis update: a novel classification. Pattern recognition letters 24(16), 3037–3058 (2003)

20. Menten, M.J., Paetzold, J.C., Zimmer, V.A., Shit, S., Ezhov, I., Holland, R., Probst, M., Schnabel, J.A., Rueckert, D.: A skeletonization algorithm for gradientbased optimization. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 21394–21403 (2023)

21. Mnih, V.: Machine learning for aerial image labeling. University of Toronto (Canada) (2013)

22. Mosinska, A., Koziński, M., Fua, P.: Joint segmentation and path classification of curvilinear structures. IEEE transactions on pattern analysis and machine intelligence 42(6), 1515–1521 (2019)

23. Mou, L., Zhao, Y., Chen, L., Cheng, J., Gu, Z., Hao, H., Qi, H., Zheng, Y., Frangi, A., Liu, J.: Cs-net: Channel and spatial attention network for curvilinear structure segmentation. In: Medical Image Computing and Computer Assisted Intervention— MICCAI 2019: 22nd International Conference, Shenzhen, China, October 13–17, 2019, Proceedings, Part I 22. pp. 721–730. Springer (2019)

24. Mou, L., Zhao, Y., Fu, H., Liu, Y., Cheng, J., Zheng, Y., Su, P., Yang, J., Chen, L., Frangi, A.F., et al.: Cs2-net: Deep learning segmentation of curvilinear structures in medical imaging. Medical image analysis 67, 101874 (2021)

25. Palti-Wasserman, D., Brukstein, A.M., Beyar, R.P.: Identifying and tracking a guide wire in the coronary arteries during angioplasty from x-ray images. IEEE Transactions on Biomedical Engineering 44(2), 152–164 (1997)

26. Peng, Y., Pan, L., Luan, P., Tu, H., Li, X.: Curvilinear object segmentation in medical images based on odos filter and deep learning network. arXiv preprint arXiv:2301.07475 (2023)

27. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: Medical Image Computing and Computer-Assisted Intervention-MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. pp. 234–241. Springer (2015)

28. Rougé, P., Passat, N., Merveille, O.: Cascaded multitask u-net using topological loss for vessel segmentation and centerline extraction. arXiv preprint arXiv:2307.11603 (2023)

29. Roychowdhury, S., Koozekanani, D.D., Parhi, K.K.: Iterative vessel segmentation of fundus images. IEEE Transactions on Biomedical Engineering 62(7), 1738–1749 (2015)

30. Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., et al.: Imagenet large scale visual recognition challenge. International journal of computer vision 115, 211–252 (2015)

31. Shit, S., Paetzold, J.C., Sekuboyina, A., Ezhov, I., Unger, A., Zhylka, A., Pluim, J.P., Bauer, U., Menze, B.H.: cldice-a novel topology-preserving loss function for tubular structure segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16560–16569 (2021)

32. Staal, J., Abràmoff, M.D., Niemeijer, M., Viergever, M.A., Van Ginneken, B.: Ridge-based vessel segmentation in color images of the retina. IEEE transactions on medical imaging 23(4), 501–509 (2004)

33. Steger, C.: Extracting curvilinear structures: A differential geometric approach. In: Computer Vision—ECCV'96: 4th European Conference on Computer Vision Cambridge, UK, April 15–18, 1996 Proceedings, Volume I 4. pp. 630–641. Springer (1996)

34. Subirats, P., Dumoulin, J., Legeay, V., Barba, D.: Automation of pavement surface crack detection using the continuous wavelet transform. In: 2006 International Conference on Image Processing. pp. 3037–3040. IEEE (2006)

35. Sun, K., Zhao, Y., Jiang, B., Cheng, T., Xiao, B., Liu, D., Mu, Y., Wang, X., Liu, W., Wang, J.: High-resolution representations for labeling pixels and regions. arXiv preprint arXiv:1904.04514 (2019)

36. Tomaszkiewicz, K., Owerko, T.: A pre-failure narrow concrete cracks dataset for engineering structures damage classification and segmentation. Scientific Data 10(1), 925 (2023)

37. Viti, M., Talbot, H., Abdallah, B., Perot, E., Gogin, N.: Coronary artery centerline tracking with the morphological skeleton loss. In: 2022 IEEE International Conference on Image Processing (ICIP). pp. 2741–2745. IEEE (2022)

38. Van der Walt, S., Schönberger, J.L., Nunez-Iglesias, J., Boulogne, F., Warner, J.D., Yager, N., Gouillart, E., Yu, T.: scikit-image: image processing in python. PeerJ 2, e453 (2014)

39. Wang, F., Gu, Y., Liu, W., Yu, Y., He, S., Pan, J.: Context-aware spatio-recurrent curvilinear structure segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12648–12657 (2019)

40. Yang, K., Musio, F., Ma, Y., Juchler, N., Paetzold, J.C., Al-Maskari, R., Höher, L., Li, H.B., Hamamci, I.E., Sekuboyina, A., et al.: Benchmarking the cow with the topcow challenge: Topology-aware anatomical segmentation of the circle of willis for cta and mra. arXiv preprint arXiv:2312.17670 (2023)

41. Zana, F., Klein, J.C.: Segmentation of vessel-like patterns using mathematical morphology and curvature evaluation. IEEE transactions on image processing 10(7), 1010–1019 (2001)

42. Zhang, T.Y., Suen, C.Y.: A fast parallel algorithm for thinning digital patterns. Communications of the ACM 27(3), 236–239 (1984)

43. Zijdenbos, A.P., Dawant, B.M., Margolin, R.A., Palmer, A.C.: Morphometric analysis of white matter lesions in mr images: method and validation. IEEE transactions on medical imaging 13(4), 716–724 (1994)