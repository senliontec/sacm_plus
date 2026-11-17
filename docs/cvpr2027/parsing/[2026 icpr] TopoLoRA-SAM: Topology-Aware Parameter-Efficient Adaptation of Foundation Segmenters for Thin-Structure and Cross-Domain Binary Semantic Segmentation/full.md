![](images/1b3e356146cbd119fe1d580d80e10df8e54d4b16949aab6b03c607cd0a926d2f.jpg)

# TopoLoRA-SAM: Topology-Aware Parameter-Efficient Adaptation of Foundation Segmenters for Thin-Structure and Cross-Domain Binary Semantic Segmentation

Salim Khazem¹

1Talan Research & Innovation Center, Paris, France

Foundation segmentation models such as the Segment Anything Model (SAM) exhibit strong zero-shot generalization through large-scale pretraining, but adapting them to domain-specific semantic segmentation remains challenging, particularly for thin structures (e.g., retinal vessels) and noisy modalities (e.g., SAR imagery). Full fine-tuning is computationally expensive and risks catastrophic forgetting. We propose TopoLoRA-SAM, a topology-aware and parameter-efficient adaptation framework for binary semantic segmentation. TopoLoRA-SAM injects Low-Rank Adaptation (LoRA) into the frozen ViT encoder, augmented with a lightweight spatial convolutional adapter and optional topology-aware supervision via differentiable clDice. We evaluate our approach on five benchmarks spanning retinal vessel segmentation (DRIVE, STARE, CHASE DB1), polyp segmentation (Kvasir-SEG), and SAR sea/- land segmentation (SL-SSDD), comparing against U-Net, DeepLabV3+, SegFormer, and Mask2Former. TopoLoRA-SAM achieves the best retina-average Dice and the best overall average Dice across datasets, while training only 5.2% of model parameters (\~4.9M). On the challenging CHASE DB1 dataset, our method substantially improves segmentation accuracy and robustness, demonstrating that topologyaware parameter-efficient adaptation can match or exceed fully fine-tuned specialist models. Code is available at : https://github.com/salimkhazem/Seglab.git

Keywords: semantic segmentation , parameter-efficient fine-tuning , topology-aware loss , Segment Anything Model , cross-domain

Date: January 6, 2026

Correspondence: salim.khazem@talan.com

Code: https://github.com/salimkhazem/Seglab.git

ORCID: 0000-0001-5958-6120

## 1 Introduction

Semantic segmentation underpins numerous critical applications in medical imaging and remote sensing, from automated diabetic retinopathy screening [1] to maritime surveillance and environmental monitoring [2] While recent foundation models such as the Segment Anything Model (SAM) [3] have achieved remarkable zero-shot generalization through web-scale pretraining, significant challenges remain when adapting these models to specialized domains with distinct visual characteristics. Two challenges are particularly acute in this setting. First, thin and elongated structures such as retinal vasculature or road networks exhibit extreme aspect ratios, where small pixel-level errors can disconnect entire branches; standard region-based losses (e.g., Dice, cross-entropy) are largely insensitive to such topological violations [4, 5]. Second, cross-domain visual shift arises in modalities such as SAR or fundoscopy, whose texture, noise characteristics, and semantics differ substantially from natural images, limiting the transferability of web-pretrained representations [6].

Full fine-tuning of billion-parameter foundation models is computationally prohibitive and risks catastrophic forgetting of generalizable representations [7]. Parameter-efficient fine-tuning (PEFT) methods such as LoRA [7], adapters [8, 9], and visual prompt tuning [10] offer compelling alternatives by updating only a small fraction of parameters while preserving pretrained knowledge. Recent work has begun exploring PEFT for SAM adaptation [11, 12, 13], yet systematic studies investigating topology-aware adaptation for domain-specific binary segmentation remain limited.

![](images/382471a25562688aad47bca40ae9164ef2b278ea4e43dbd056df14264c497e61.jpg)  
Figure 1 TopoLoRA-SAM architecture overview. We freeze the SAM ViT-B image encoder and inject LoRA modules (red) into the feed-forward network (FFN) layers of each transformer block. A lightweight depthwise-separable convolutional adapter (green) refines the high-resolution embedding tensor before mask decoding. Training uses a topology-aware loss combining BCE, Dice, and clDice.

In this work, we introduce TopoLoRA-SAM, a principled framework that unifies parameter-efficient adaptation with topology-preserving supervision for binary semantic segmentation across diverse domains. Our approach freezes the SAM ViT-B image encoder and injects trainable LoRA modules into its feed-forward layers, complemented by a lightweight depthwise-separable convolutional adapter operating on high-resolution feature maps. This design enables fine-grained adaptation while preserving the generalizable representations learned during pretraining. To address topological integrity critical for thin structures we incorporate the differentiable clDice loss [4], which explicitly optimizes skeleton overlap and encourages connected centerlines. Our contributions are: (i) a topology-aware, parameter-efficient adaptation of SAM combining LoRA and a lightweight spatial adapter; (ii) a benchmark across five binary segmentation datasets (retinal vessels polyps, SAR sea/land) with region, boundary, topology, and calibration metrics; and (iii) strong performance on CHASE\_DB1 and best retina-average Dice while training only 5.2% of model parameters, with a fully reproducible codebase. In this work, ideas related to simplified geometric representations for efficient visual learning [14] and lightweight adaptation via visual prompting [15] are leveraged within a foundation-model segmentation framework.

## 2 Related Work

Semantic segmentation architectures. Fully convolutional networks revolutionized semantic segmentation, with U-Net [16] establishing the encoder-decoder paradigm with skip connections that remains foundational in medical imaging. DeepLabV3+ [17| introduced atrous spatial pyramid pooling for multi-scale context aggregation, while recent self-configuring approaches like nnU-Net [18| achieve strong performance through systematic hyperparameter optimization. [19, 20] used these kind of architectures for different applications. Transformer-based architectures have emerged as powerful alternatives: SegFormer [21] proposes an efficient hierarchical design with lightweight MLP decoders, while Mask2Former [22]| unifies instance, semantic, and panoptic segmentation through masked attention. These architectures build upon foundational work on vision transformers [23] and hierarchical designs [24].

Foundation models for segmentation. The Segment Anything Model [3] represents a paradigm shift, trained on over 1 billion masks to enable promptable zero-shot segmentation. While SAM excels at interactive mask generation, adapting it to semantic segmentation particularly for domain-specific binary masks without prompts requires careful consideration of the prompt-mask relationship. SAM 2 [25] extends this capability to video, demonstrating the scalability of the prompting paradigm. However, for domains with significant distribution shift (medical imaging, remote sensing), prompt-free adaptation with task-specific tuning remains essential [26].

SAM adaptation for specialized domains. Several concurrent works address SAM adaptation for medical imaging. MedSAM [12] fine-tunes SAM on large-scale medical datasets, demonstrating strong generalization across modalities. SAM-Adapter [11] proposes domain-specific adapter modules for underperformed scenes, while SAM-Med2D [27] focuses on 2D medical image segmentation. HQ-SAM [28] introduces a high-quality output token for improved mask boundaries. PerSAM [29] enables one-shot personalization through trainingfree prompt learning. Our work differs by jointly addressing parameter efficiency (LoRA + adapters) and topology preservation (clDice) for thin-structure domains where connectivity is critical.

Parameter-efficient fine-tuning. PEFT methods enable foundation model adaptation with minimal parameter overhead. LoRA [7] reparameterizes weight updates as low-rank matrices, achieving comparable performance to full fine-tuning on language models. AdaLoRA [30] extends this with adaptive rank allocation, while DoRA [31] decomposes weights into magnitude and direction for improved learning dynamics. Adapter modules [8] insert trainable bottleneck layers between transformer blocks, and AdaptFormer [9] adapts this for vision. Visual Prompt Tuning (VPT) [10] prepends learnable tokens to input sequences. Scale-and-shift tuning (SSF) [32] offers an even more parameter-efficient alternative. These methods share the goal of preserving pretrained knowledge while enabling task-specific adaptation.

Topology-aware losses and metrics. Standard overlap metrics (Dice [33], IoU) are insensitive to topological errors that break connectivity in thin structures. clDice [4] addresses this by computing skeleton overlap using differentiable soft-skeletonization, encouraging connected centerlines in tubular structures. Complementary approaches include TopoLoss [5], which uses persistent homology to penalize Betti number mismatches, and Betti matching [34], which aligns persistence barcodes for topologically faithful predictions. Boundary focused metrics such as BFScore [35] evaluate contour precision. For probabilistic outputs, calibration metrics like Expected Calibration Error (ECE) [36, 37] assess reliability, which is critical for clinical deployment [38]. Recent work emphasizes the importance of comprehensive metric evaluation [39].

## 2.1 Problem Formulation

Given an input image $\mathbf { x } \in \mathbb { R } ^ { H \times W \times 3 }$ , we aim to predict a binary semantic mask $\hat { \mathbf { y } } \in [ 0 , 1 ] ^ { H \times W }$ indicating foreground probability (vessel, polyp, or sea regions). Unlike SAM's promptable paradigm that produces instance-agnostic masks conditioned on user-provided points or boxes, our task requires semantic predictions without runtime prompts. This necessitates adapting SAM's encoder-decoder architecture to directly output semantic masks through end-to-end training.

## 2.2 SAM Architecture and Prompt-Free Decoding

We adopt the SAM ViT-B architecture [3], comprising: (i) a Vision Transformer (ViT) image encoder with 12 transformer blocks, producing embeddings $\mathbf { z } \in \mathbb { R } ^ { 2 5 6 \times \frac { H } { 1 6 } \times \frac { W } { 1 6 } }$ ; (ii) a prompt encoder mapping user interactions to embedding space; and (iii) a lightweight mask decoder that combines image and prompt embeddings via cross-attention. For semantic segmentation, we operate in prompt-free mode by providing null prompt embeddings (∅), treating the mask decoder as a semantic prediction head. The decoder outputs a single binary mask, upsampled to the target resolution via bilinear interpolation.

## 2.3 Low-Rank Adaptation (LoRA)

To enable efficient adaptation while preserving pretrained representations, we inject LoRA modules [7] into the frozen image encoder. For a pretrained linear layer with weight matrix $\mathbf { W } _ { 0 } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ , LoRA reparameterizes the forward pass as:

$$
\begin{array} { r } { \mathbf { h } = \mathbf { W } _ { 0 } \mathbf { x } + \Delta \mathbf { W } \mathbf { x } = \mathbf { W } _ { 0 } \mathbf { x } + \mathbf { B } \mathbf { A } \mathbf { x } , } \end{array}\tag{1}
$$

where $\mathbf { A } \in \mathbb { R } ^ { r \times d _ { \mathrm { i n } } }$ and $\mathbf { B } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times r }$ are trainable low-rank factors with rank $r \ll \operatorname* { m i n } ( d _ { \mathrm { i n } } , d _ { \mathrm { o u t } } )$ . We initialize A with Kaiming uniform and B with zeros, ensuring $\Delta \mathbf { W } = \mathbf { 0 }$ at initialization for stable training. A scaling factor $\alpha / r$ modulates the magnitude of low-rank updates.

We target the feed-forward network (FFN) layers within each transformer block, specifically the mlp.1in1 and mlp.lin2 projections. This design choice follows the intuition that FFN layers capture task-specific feature transformations, while attention weights encode more transferable relational patterns. With r = 16 and LoRA applied to all 12 blocks, we add approximately 2.4M trainable parameters to the encoder.

Table 1 Parameter efficiency comparison. TopoLoRA-SAM trains only 5.2% of parameters while achieving competitive or superior performance.
<table><tr><td>Model</td><td>Total (M)</td><td>Trainable (M)</td><td>Trainable %</td></tr><tr><td>U-Net (ResNet34)</td><td>24.4</td><td>24.4</td><td>100.0%</td></tr><tr><td>DeepLabV3+ (ResNet50)</td><td>39.8</td><td>39.8</td><td>100.0%</td></tr><tr><td>SegFormer (MiT-B0)</td><td>3.7</td><td>3.7</td><td>100.0%</td></tr><tr><td>Mask2Former (Swin-T)</td><td>47.4</td><td>47.4</td><td>100.0%</td></tr><tr><td>TopoLoRA-SAM (ours)</td><td>93.7</td><td>4.9</td><td>5.2%</td></tr></table>

## 2.4 Spatial Convolutional Adapter

While LoRA enables semantic adaptation of the encoder's representational capacity, thin structures require fine-grained spatial reasoning at higher resolutions than SAM's 16× downsampled embeddings. We introduce a lightweight depthwise-separable convolutional adapter applied to the image embedding tensor z:

$$
\mathbf { z } ^ { \prime } = \mathbf { z } + \operatorname { C o n v } _ { 1 \times 1 } \left( \operatorname { R e L U } \left( \operatorname { D e p t h w i s e C o n v } _ { 3 \times 3 } ( \mathbf { z } ) \right) \right) .\tag{2}
$$

The adapter comprises a $3 \times 3$ depthwise convolution (256 groups) followed by ReLU activation and $1 \times 1$ pointwise projection, wrapped in a residual connection. This adds only \~66K parameters while enabling local spatial refinement crucial for thin-structure boundary preservation.

## 2.5 Topology-Aware Training Objective

Our training objective combines multiple complementary loss terms:

$$
\mathcal { L } = \lambda _ { \mathrm { b c e } } \mathcal { L } _ { \mathrm { B C E } } + \lambda _ { \mathrm { d i c e } } \mathcal { L } _ { \mathrm { D i c e } } + \lambda _ { \mathrm { c l } } \mathcal { L } _ { \mathrm { c l D i c e } } + \lambda _ { \mathrm { b d } } \mathcal { L } _ { \mathrm { B o u n d a r y } } ,\tag{3}
$$

with default weights $\lambda _ { \mathrm { b c e } } = 1 . 0 , \lambda _ { \mathrm { d i c e } } = 1 . 0 , \lambda _ { \mathrm { c l } } = 0 . 5 , \mathrm { a n d } \lambda _ { \mathrm { b d } } = 0 . 0 .$

Binary cross-entropy (BCE) provides pixel-wise gradient signal:

$$
\mathcal { L } _ { \mathrm { B C E } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ y _ { i } \log ( \hat { y } _ { i } ) + ( 1 - y _ { i } ) \log ( 1 - \hat { y } _ { i } ) \right] .\tag{4}
$$

Soft Dice loss optimizes region overlap:

$$
\mathcal { L } _ { \mathrm { D i c e } } = 1 - \frac { 2 \sum _ { i } y _ { i } \hat { y } _ { i } + \epsilon } { \sum _ { i } y _ { i } + \sum _ { i } \hat { y } _ { i } + \epsilon } .\tag{5}
$$

Centerline Dice (clDice) [4] enforces topological connectivity by computing Dice on soft-skeletonized predictions:

$$
\mathcal { L } _ { \mathrm { c l D i c e } } = 1 - 2 \cdot \frac { \mathrm { T p r e c } ( \hat { y } , y ) \cdot \mathrm { T s e n s } ( \hat { y } , y ) } { \mathrm { T p r e c } ( \hat { y } , y ) + \mathrm { T s e n s } ( \hat { y } , y ) } ,\tag{6}
$$

where Tprec measures how much of the predicted skeleton overlaps with the ground-truth mask, and Tsens measures coverage of the GT skeleton. Soft skeletonization is implemented via iterated morphological operations (min-pooling) applied to the soft prediction map.

## 2.6 Implementation Details

We use SAM ViT-B pretrained weights and freeze all encoder parameters except LoRA modules. The mask decoder remains trainable (\~2.4M parameters). Total trainable parameters: \~4.9M (LoRA: 2.4M, adapter: 66K, decoder: 2.4M), compared to 93.7M total. Table 1 summarizes parameter efficiency. Training uses AdamW with learning rate $1 0 ^ { - 4 }$ , cosine decay, and 50 epochs

## 3 Experimental Setup

## 3.1 Datasets

We evaluate on five binary segmentation benchmarks spanning three distinct application domains:

Retinal vessel segmentation. DRIVE [1] contains 40 color fundus images (584×565) with manual vessel annotations from expert ophthalmologists, split into 20 train and 20 test images. STARE [40] provides 20 images (700×605) with vessels and pathological abnormalities. CHASE\_DB1 [41] comprises 28 highresolution images (999×960) from a pediatric population study. These datasets exemplify thin-structure segmentation where topological connectivity is clinically significant for vascular analysis.

Polyp segmentation. Kvasir-SEG [42] contains 1,000 colonoscopy images with corresponding polyp masks, representing an important task for early colorectal cancer detection. Images vary in resolution and polyp appearance.

SAR sea/land segmentation. SL-SSDD [2] provides synthetic aperture radar (SAR) imagery for maritime boundary delineation, challenging due to speckle noise, low contrast, and significant distribution shift from natural images.

## 3.2 Preprocessing and Augmentations

All datasets are standardized to RGB format with ImageNet normalization $( \mu = [ 0 . 4 8 5 , 0 . 4 5 6 , 0 . 4 0 6 ] , \sigma =$ [0.229, 0.224, 0.225]). We use dataset-specific resolutions: 384×384 for retinal datasets, 512×512 for Kvasir-SEG and SL-SSDD. Training augmentations include random horizontal/vertical flips, random scaling (0.5– 1.5×), and center crops with reflection padding. Color jitter is disabled for SAR imagery to preserve modalityspecific characteristics.

## 3.3 Baseline Architectures

We compare five segmentation model families spanning convolutional, transformer-based, and foundationmodel paradigms. As convolutional baselines, we use U-Net [16] with a ResNet34 [43] encoder and DeepLabV3+ [17 with a ResNet50 encoder. For transformer-based segmentation, we evaluate SegFormer (MiT-B0) [21] and Mask2Former [22] with a Swin-T backbone [24]. Finally, we include our TopoLoRA-SAM model, which adapts SAM ViT-B [3] using Low-Rank Adaptation, a lightweight spatial adapter, and optional topologyaware clDice regularization. All models use ImageNet-pretrained backbones (or SAM-pretrained weights) and are trained with an identical BCE+Dice objective to ensure fair comparison.

## 3.4 Training Configuration

All experiments follow a unified training protocol. We use the AdamW optimizer with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ and a weight decay of $1 0 ^ { - 4 }$ . The learning rate is set to $1 0 ^ { - 4 }$ and scheduled using cosine annealing down to a minimum value of $1 0 ^ { - 6 }$ . All models are trained for 50 epochs. A batch size of 4 is used for all baseline models, while SAM-based models are trained with a batch size of 1 and gradient accumulation over 4 steps to maintain a comparable effective batch size. Training is performed using mixed-precision (FP16) via PyTorch Automatic Mixed Precision (AMP). For robustness, all reported results are averaged over three random seeds (0, 1, and 2), and we report the mean and standard deviation.

## 3.5 Evaluation Metrics

We use a comprehensive evaluation suite covering region overlap, boundary accuracy, topology preservation, and calibration. Region-level performance is measured using the Dice coefficient (F1-score) and Intersectionover-Union(IoU), with precision and recall reported to characterize false positive and false negative behavior. Boundary quality is evaluated using the BFScore [35], which computes an F1-score on contour pixels within a tolerance margin. To assess topology preservation for thin structures, we report centerline Dice (clDice) [4] on retinal datasets, capturing connectivity via skeleton overlap. Finally, probabilistic calibration is measured using Expected Calibration Error (ECE) [36], which quantifies the alignment between prediction confidence and empirical accuracy; lower values indicate better-calibrated models, an important consideration for clinical decision support.

Table 2 Benchmark Dice scores (mean±std across 3 seeds). Bold indicates best performance per dataset. TopoLoRA-SAM achieves the best overall average and best retina-average Dice while training only 5.2% of parameters.
<table><tr><td></td><td>U-Net (R34)</td><td>DeepLabV3+ (R50)</td><td>SegFormer (B0)</td><td>Mask2Former (Swin-T)</td><td>TopoLoRA-SAM (ours)</td></tr><tr><td>DRIVE</td><td>0.590±0.059</td><td>0.589±0.027</td><td>0.541±0.041</td><td>0.689±0.011</td><td>0.690±0.018</td></tr><tr><td>STARE</td><td>0.433±0.131</td><td>0.478±0.056</td><td>0.374±0.016</td><td>0.644±0.023</td><td>0.565±0.048</td></tr><tr><td>CHASE_DB1</td><td>0.412±0.050</td><td>0.345±0.123</td><td>0.424±0.011</td><td>0.285±0.293</td><td>0.569±0.016</td></tr><tr><td>Kvasir-SEG</td><td>0.922±0.004</td><td>0.917±0.003</td><td>0.920±0.005</td><td>0.933±0.012</td><td>0.930±0.002</td></tr><tr><td>SL-SSDD</td><td>0.993±0.001</td><td>0.993±0.000</td><td>0.994±0.000</td><td>0.992±0.001</td><td>0.994±0.000</td></tr><tr><td>Retina Avg</td><td>0.478±0.113</td><td>0.470±0.126</td><td>0.446±0.077</td><td>0.539±0.242</td><td>0.595±0.049</td></tr><tr><td>Overall Avg</td><td>0.670±0.259</td><td>0.664±0.265</td><td>0.650±0.267</td><td>0.709±0.283</td><td>0.735±0.185</td></tr></table>

## 3.6 Computational Resources

All experiments run on a single NVIDIA RTX A6000 ada (50GB) GPU per training job, with multi-GPU parallelism used only to run independent experiments concurrently. The complete benchmark (5 datasets × 5 models × 3 seeds + ablations + cross-dataset) requires approximately 29.7 GPU-hours, dominated by Mask2Former and SAM runs due to larger memory footprint and longer convergence.

## 4 Results

## 4.1 Main Benchmark Results

Table 2 reports Dice scores across all datasets and model families. Overall, TopoLoRA-SAM achieves the best average Dice (0.735), outperforming the second-best Mask2Former (0.709) while training only 4.9M parameters, compared to 47.4M for Mask2Former. This highlights the effectiveness of parameter-efficient adaptation of SAM representations.

On retinal vessel segmentation, TopoLoRA-SAM shows consistent and robust performance. It achieves the best Dice on CHASE\_DB1 (0.569) with substantially lower variance across seeds, corresponding to a +8.4 Dice improvement over Mask2Former. On DRIVE and STARE, TopoLoRA-SAM remains competitive with transformer-based baselines and achieves the best retina-average Dice (0.595), indicating stable generalization across datasets with different image characteristics.

Across non-retinal domains, TopoLoRA-SAM maintains strong cross-domain behavior, achieving competitive Dice on Kvasir-SEG (0.900) and matching state-of-the-art performance on SL-SSDD (0.993), despite the significant distribution shift from natural images. These results suggest that SAM representations adapted via LoRA transfer effectively across heterogeneous imaging modalities.

## 4.2 Topology Preservation

For thin-structure segmentation, region overlap metrics alone are insufficient, as topological connectivity is critical for downstream analysis. Figure 2 summarizes clDice and BFScore on retinal datasets. TopoLoRA-SAM achieves the best cIDice on DRIVE (0.678) and CHASE\_DB1 (0.599), indicating improved preservation of vessel connectivity, particularly on high-resolution images where baseline methods often produce fragmented predictions. Improvements are most pronounced on CHASE DB1, with a gain of over 6 clDice points compared to the second-best method.

Boundary quality follows a similar trend. TopoLoRA-SAM achieves the best retina-average BFScore (0.578), with strong gains on CHASE\_DB1, highlighting its ability to delineate thin vessel boundaries accurately an important property for vessel diameter estimation and stenosis analysis.

![](images/784321baa282f83b8c8359267b08ec506eec88fb2545d92ef7558ed2cf624c9c.jpg)

![](images/7372b6917e877f7a98ad6c397808b55e9dd43671e79bf2068471fe786e0d84c9.jpg)

![](images/bbc5b7c065afa16f89aea840ab51632e16d4dc6d531978e4e4e7f2214798327e.jpg)  
Figure 2 Thin-structure performance summary across retinal datasets (DRIVE, STARE, CHASE DB1). TopoLoRA-SAM achieves top-tier performance on Dice, clDice, and BFScore while training 10× fewer parameters.

## 4.3 Parameter Efficiency Analysis

Fig 4 visualizes the fundamental trade-off between model capacity and performance on retinal datasets. Key observations:

• TopoLoRA-SAM achieves Pareto-optimal performance, offering the best Dice among methods with <10M trainable parameters.

• Compared to fully-tuned Mask2Former (47.4M params), TopoLoRA-SAM (4.9M params) achieves superior retina-average Dice with 90% parameter reduction.

• The lightweight SegFormer (3.7M) shows the lowest parameter count but significantly underperforms on thin-structure preservation.

This analysis demonstrates that parameter-efficient adaptation of SAM's powerful pretrained representations is a compelling alternative to training specialist segmentation architectures from scratch.

## 4.4 Calibration Analysis

Reliable confidence estimates are essential for risk-aware decision support. Across all datasets, TopoLoRA-SAM demonstrates strong calibration, achieving the lowest Expected Calibration Error (ECE) on CHASE\_- DB1 (0.045) and SL-SSDD (0.003), highlighting its robustness in both medical imaging and remote sensing scenarios. While SegFormer attains the best average ECE overall, this comes at the expense of reduced segmentation accuracy. In contrast, Mask2Former exhibits pronounced miscalibration on CHASE DB1, consistent with its high Dice score variability. Overall, TopoLoRA-SAM offers a favorable balance between segmentation accuracy and probabilistic calibration.

## 4.5 Component Contributions

LoRA adaptation is the primary driver. Adding LoRA modules to the frozen encoder provides the largest single improvement across all metrics: +4.2 Dice, +5.1 clDice, and +8.3 BFScore points over decoder-only tuning. This confirms that adapting the encoder's feature representations is critical for domain-specific segmentation, even with a small parameter budget (2.4M additional parameters).

Spatial adapter provides boundary refinement. The depthwise-separable adapter offers modest improvements on boundary-sensitive metrics (+1.2 BFScore over LoRA-only), suggesting that high-resolution spatial refinement benefits thin-structure delineation. However, gains are dataset-dependent and may require tuning for optimal integration.

Topology regularization benefits connectivity. Adding clDice loss improves skeleton-based metrics (clDice) by +0.8 points on average, with the effect most pronounced on images with complex vessel branching. How-

U-Net

![](images/8e30124bee19af6e82bf1ad9cc44432c69229a7169d29c19fb75a864d3eb4149.jpg)  
Figure 3 Qualitative comparison across all benchmark datasets. Each row shows a representative sample from one dataset: retinal vessel segmentation (DRIVE, STARE, CHASE\_DB1), polyp segmentation (Kvasir-SEG), and SAR sea/land segmentation (SL-SSDD). Columns display input images, ground truth masks, and predictions from five methods. TopoLoRA-SAM (Ours) demonstrates superior preservation of thin vessel structures and fine-grained boundaries compared to baselines, particularly visible in the retinal datasets where thin peripheral vessels are better captured. On CHASE\_ DB1, our method maintains vessel connectivity while other methods show fragmentation. Best viewed zoomed in.

ever, the improvement is sensitive to the regularization weight $\lambda _ { \mathrm { c l } } \mathrm { : }$ too high values can destabilize training, while too low values provide negligible benefit

## 4.6 LoRA Rank Sensitivity

We ablate the LoRA rank $r \in \{ 4 , 8 , 1 6 , 3 2 \}$ on DRIVE (fixing other components):
<table><tr><td>Rank</td><td>Params (M)</td><td>Dice</td><td>clDice</td></tr><tr><td>r = 4</td><td>0.6</td><td> $0 . 6 3 2 { \pm } 0 . 0 2 1$ </td><td>0.659±0.024</td></tr><tr><td>r = 8</td><td>1.2</td><td> $0 . 6 4 1 { \pm } 0 . 0 1 9$ </td><td> $0 . 6 6 8 { \pm } 0 . 0 2 2$ </td></tr><tr><td>r = 16</td><td>2.4</td><td> $\mathbf { 0 . 6 9 0 { \pm } } 0 . 0 1 8$ </td><td> $\mathbf { 0 . 6 7 8 { \scriptstyle \pm 0 . 0 2 1 } }$ </td></tr><tr><td>r = 32</td><td>4.8</td><td> $0 . 6 4 8 { \pm } 0 . 0 2 0$ </td><td> $0 . 6 7 5 { \scriptstyle \pm 0 . 0 2 3 }$ </td></tr></table>

Rank $r = 1 6$ offers the best balance of capacity and efficiency. Higher ranks $( r = 3 2 )$ provide diminishing returns while doubling parameter count, consistent with observations from LoRA [7] in language models.

![](images/5bf9922c1305f418734b0714a175638c444116a647d4cf3cc69c47106ae02546.jpg)  
Figure 4 Parameter efficiency trade-off on retinal datasets. TopoLoRA-SAM achieves Pareto-optimal performance: competitive Dice with 10× fewer trainable parameters than Mask2Former.

## 4.7 Loss Weight Sensitivity

We vary the clDice weight $\lambda _ { \mathrm { c l } } \in \{ 0 . 0 , 0 . 2 5 , 0 . 5 , 1 . 0 , 2 . 0 \}$

<table><tr><td> $\lambda _ { \mathrm { c l } }$ </td><td>Dice</td><td>clDice</td><td>ECE</td></tr><tr><td>0.0</td><td> $0 . 6 4 8 { \pm } 0 . 0 1 9$ </td><td> $0 . 6 7 1 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $0 . 0 4 1 { \pm } 0 . 0 0 8$ </td></tr><tr><td>0.25</td><td> $0 . 6 4 9 { \pm } 0 . 0 1 8$ </td><td> $0 . 6 7 5 { \scriptstyle \pm 0 . 0 2 1 }$ </td><td> $0 . 0 4 2 { \scriptstyle \pm 0 . 0 0 9 }$ </td></tr><tr><td>0.5</td><td> $\mathbf { 0 . 6 9 0 { \overset { . } { \pm } } 0 . 0 1 8 }$ </td><td> $\mathbf { 0 . 6 7 8 { \scriptstyle \pm 0 . 0 2 1 } }$ </td><td> $0 . 0 4 3 { \pm } 0 . 0 0 9$ </td></tr><tr><td>1.0</td><td> $0 . 6 4 7 { \scriptstyle \pm 0 . 0 2 0 }$ </td><td> $0 . 6 7 6 { \scriptstyle \pm 0 . 0 2 3 }$ </td><td> $0 . 0 4 6 { \pm } 0 . 0 1 0$ </td></tr><tr><td>2.0</td><td> $0 . 6 3 8 { \pm } 0 . 0 2 4$ </td><td> $0 . 6 7 0 { \scriptstyle \pm 0 . 0 2 7 }$ </td><td> $0 . 0 5 2 { \scriptstyle \pm 0 . 0 1 2 }$ </td></tr></table>

Moderate topology weighting $( \lambda _ { \mathrm { c l } } = 0 . 5 )$ achieves optimal balance. Excessive weighting $\left( \lambda _ { \mathrm { c l } } \ge 2 . 0 \right)$ degrades both region and topology metrics, likely due to gradient conflicts between objectives.

Our ablation results suggest several practical defaults. When compute or turnaround time is constrained, LoRA in encoder FFNs (rank $r = 1 6 )$ accounts for most of the improvement with minimal overhead. In applications where connectivity is a primary objective, clDice regularization with $\lambda _ { \mathrm { c l } } = 0 . 5$ is a plausible choice, although its benefit is dataset-dependent and should be checked on held-out data. For tasks requiring precise boundaries (e.g., vessel thickness measurements), adding the spatial adapter can further improve contour fidelity.

## 5 Conclusion, Limitations, and Broader Impact

We introduced TopoLoRA-SAM, a topology-aware and parameter-efficient adaptation framework for the Segment Anything Model (SAM) targeting binary semantic segmentation across diverse domains. Our approach combines Low-Rank Adaptation (LoRA) in the frozen ViT encoder, a lightweight spatial convolu tional adapter, and optional topology-aware supervision via differentiable clDice. Across five benchmarks spanning retinal vessels, polyp segmentation, and SAR imagery, TopoLoRA-SAM achieves the best retinaaverage Dice and the best overall average Dice among evaluated methods, while training only 5.2% of model parameters. Notably, on the challenging CHASE\_DB1 dataset, our method improves Dice by up to +8.4 points with substantially lower variance, highlighting the robustness of parameter-efficient adaptation. Topology-focused metrics (clDice and BFScore) further confirm improved connectivity preservation for thin structures.

Our ablation studies indicate that LoRA is the primary driver of performance gains, while the spatial adapter and topology-aware loss provide complementary but more sensitive benefits. In particular, topology regularization depends on careful weighting and does not universally improve cross-dataset generalization. Moreover, while TopoLoRA-SAM trains few parameters, the frozen SAM backbone still incurs non-trivial memory and inference costs, which may limit deployment in resource-constrained settings. This work is intended as a research contribution and does not constitute a validated clinical or operational system. We encourage responsible use, especially in medical and remote sensing contexts, and release our codebase and trained models to support reproducibility and further research. Promising future directions include extension to video and 3D segmentation, multi-class settings, and domain-adaptive topology regularization

## References

[1] J. Staal, M. D. Abràmoff, M. Niemeijer, M. A. Viergever, and B. van Ginneken, "Ridge-based vessel segmentation in color images of the retina," IEEE Transactions on Medical Imaging, vol. 23, no. 4, pp. 501–509, 2004.

[2] S. Wang, W. Guo, Z. Cui, and W. Shi, "Sea-land segmentation in sar images using supervised contrastive learning and transformer," IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 15, pp. 8900–8913, 2022.

[3] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. Berg, W.-Y. Lo, P. Dollár, and R. Girshick, "Segment anything," in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4015–4026, 2023.

[4] S. Shit, J. C. Paetzold, A. Sekuboyina, I. Ezhov, A. Unger, A. Zhylka, N. Thuerey, F. Alongi, and B. H. Menze, "cldice – a novel topology-preserving loss function for tubular structure segmentation," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 16560–16569, 2021.

[5] X. Hu, F. Li, D. Samaras, and C. Chen, "Topology-preserving deep image segmentation," in Advances in Neural Information Processing Systems (NeurIPS), vol. 32, 2019.

[6] C. Chen, Q. Dou, H. Chen, J. Qin, and P.-A. Heng, "Domain adaptation for medical image analysis: A survey," in IEEE Transactions on Medical Imaging, vol. 41, pp. 1926–1950, 2022.

[7] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "Lora: Low-rank adaptation of large language models," in International Conference on Learning Representations (ICLR), 2022.

[8] N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. de Laroussilhe, A. Gesmundo, M. Attariyan, and S. Gelly, "Parameter-efficient transfer learning for nlp," in International Conference on Machine Learning (ICML), pp. 2790–2799, 2019.

[9] S. Chen, C. Ge, Z. Tong, J. Wang, Y. Song, J. Wang, and P. Luo, "Adaptformer: Adapting vision transformers for scalable visual recognition," in Advances in Neural Information Processing Systems (NeurIPS), vol. 35, pp. 16664– 16678, 2022.

[10] M. Jia, L. Tang, B.-C. Chen, C. Cardie, S. Belongie, B. Hariharan, and S.-N. Lim, "Visual prompt tuning," in European Conference on Computer Vision (ECCV), pp. 709–727, 2022

[11] T. Chen, L. Zhu, C. Ding, R. Cao, Y. Wang, Z. Li, L. Sun, Y. Mao, and P. Zang, "Sam fails to segment anything? – sam-adapter: Adapting sam in underperformed scenes," in Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), pp. 3367–3375, 2023.

[12] J. Ma, Y. He, F. Li, L. Han, C. You, and B. Wang, "Segment anything in medical images," Nature Communications, vol. 15, no. 1, p. 654, 2024.

[13] X. Zhang, X. Liu, F. Guo, S. Liu, P. Ren, Y. Gao, and Y. Wang, "How to efficiently adapt large segmentation model (sam) to medical images," in Medical Image Computing and Computer Assisted Intervention (MICCAI) 2023.

[14] S. Khazem, J. Fix, and C. Pradalier, "Polygonet: Leveraging simplified polygonal representation for effective image classification," arXiv preprint arXiv:2504.01214, 2025.

[15] S. Khazem, “Multi-scale visual prompting for lightweight small-image classification," arXiv preprint arXiv:2512.03663, 2025.

[16] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation," in International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), pp. 234– 241, 2015.

[17] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, "Encoder-decoder with atrous separable convolution for semantic image segmentation," in European Conference on Computer Vision (ECCV), pp. 801–818, 2018.

[18] F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, and K. H. Maier-Hein, "nnu-net: A self-configuring method for deep learning-based biomedical image segmentation,"in Nature Methods, vol. 18, pp. 203–211, 2021.

[19] S. Khazem, A. Richard, J. Fix, and C. Pradalier, “Deep learning for the detection of semantic features in tree x-ray ct scans," Artificial Intelligence in Agriculture, vol. 7, pp. 13–26, 2023.

[20] S. Khazem, J. Fix, and C. Pradalier, "Improving knot prediction in wood logs with longitudinal feature propagation," in International Conference on Computer Vision Systems, pp. 169–180, Springer, 2023.

[21] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, "Segformer: Simple and efficient design for semantic segmentation with transformers," in Advances in Neural Information Processing Systems (NeurIPS). vol. 34, pp. 12077–12090, 2021.

[22] B. Cheng, I. Misra, A. G. Schwing, A. Kirillov, and R. Girdhar, "Masked-attention mask transformer for universal image segmentation," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1290–1299, 2022.

[23] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An image is worth 16x16 words: Transformers for image recognition at scale," in International Conference on Learning Representations (ICLR), 2021.

[24] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin transformer: Hierarchical vision transformer using shifted windows," in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 10012–10022, 2021.

[25] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson, et al., "Sam 2: Segment anything in images and videos," arXiv preprint arXiv:2408.00714, 2024.

[26] S. Khazem, L. Arnould, and H. A. Mehenni, "Byo-eval: Build your own dataset for fine-grained visual assessment of multimodal language models," 2025.

[27] J. Cheng, J. Ye, Z. Deng, J. Chen, T. Li, H. Wang, Y. Su, Z. Huang, J. Chen, L. Jiang, et al., "Sam-med2d," arXiv preprint arXiv:2308.16184, 2023.

[28] L. Ke, M. Ye, M. Danelljan, Y. Liu, Y.-W. Tai, C.-K. Tang, and F. Yu, "Segment anything in high quality," in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2024

[29] R. Zhang, Z. Jiang, Z. Guo, S. Yan, J. Pan, H. Dong, Y. Qiao, P. Gao, and H. Li, "Personalize segment anything model with one shot," arXiv preprint arXiv:2305.03048, 2023.

[30] Q. Zhang, M. Chen, A. Bukharin, P. He, Y. Cheng, W. Chen, and T. Zhao, "Adalora: Adaptive budget allocation for parameter-efficient fine-tuning," in International Conference on Learning Representations (ICLR), 2023.

[31] S.-Y. Liu, C.-Y. Wang, H. Yin, P. Molchanov, Y.-C. F. Wang, K.-T. Cheng, and M.-H. Chen, "Dora: Weightdecomposed low-rank adaptation," arXiv preprint arXiv:2402.09353, 2024.

[32] D. Lian, D. Zhou, J. Feng, and X. Wang, “Scaling & shifting your features: A new baseline for efficient model tuning," in Advances in Neural Information Processing Systems (NeurIPS), vol. 35, pp. 109–123, 2022.

[33] F. Milletari, N. Navab, and S.-A. Ahmadi, "V-net: Fully convolutional neural networks for volumetric medical image segmentation," in International Conference on 3D Vision (3DV), pp. 565–571, 2016.

[34] N. Stucki, J. C. Paetzold, S. Shit, B. Menze, and U. Bauer, "Topologically faithful image segmentation via induced matching of persistence barcodes," in International Conference on Machine Learning (ICML), pp. 32698–32727 2023.

[35] G. Csurka, D. Larlus, and F. Perronnin, "What is a good evaluation measure for semantic segmentation?," British Machine Vision Conference (BMVC), 2013.

[36] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in International Conference on Machine Learning (ICML), pp. 1321–1330, 2017.

[37] M. P. Naeini, G. F. Cooper, and M. Hauskrecht, "Obtaining well calibrated probabilities using bayesian binning," in AAAI Conference on Artificial Intelligence, pp. 2901–2907, 2015.

[38] A. Mehrtash, W. M. Wells, C. M. Tempany, P. Abolmaesumi, and T. Kapur, "Confidence calibration and predictive uncertainty estimation for deep medical image segmentation," in IEEE Transactions on Medical Imaging, vol. 39, pp. 3868–3878, 2020.

[39] A. Reinke, M. D. Tizabi, M. Baumgartner, M. Eisenmann, et al., "Understanding metric-related pitfalls in image analysis validation," Nature Methods, vol. 21, pp. 182–194, 2024.

[40] A. Hoover, V. Kouznetsova, and M. Goldbaum, "Locating blood vessels in retinal images by piecewise threshold probing of a matched filter response," IEEE Transactions on Medical Imaging, vol. 19, no. 3, pp. 203–210, 2000

[41] M. M. Fraz, P. Remagnino, A. Hoppe, B. Uyyanonvara, A. R. Rudnicka, C. G. Owen, and S. A. Barman “An ensemble classification-based approach applied to retinal blood vessel segmentation," IEEE Transactions on Biomedical Engineering, vol. 59, no. 9, pp. 2538–2548, 2012.

[42] D. Jha, P. H. Smedsrud, M. A. Riegler, P. Halvorsen, T. de Lange, D. Johansen, and H. D. Johansen, "Kvasir-seg: A segmented polyp dataset," Multimedia Tools and Applications, vol. 80, pp. 35743–35762, 2021.

[43] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778, 2016.