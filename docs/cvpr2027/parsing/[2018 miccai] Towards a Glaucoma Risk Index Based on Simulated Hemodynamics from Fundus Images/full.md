# Towards a glaucoma risk index based on simulated hemodynamics from fundus images

José Ignacio Orlando¹, João Barbosa Breda2, Karel van Keer2, Matthew B. Blaschko3, Pablo J. Blanco4, and Carlos A. Bulant1

1 CONICET - Pladema Institute, UNICEN, Argentina

2 Research Group Ophthalmology, KU Leuven, Leuven, Belgium 3 ESAT-PSI, KU Leuven, Leuven, Belgium

4 National Laboratory for Scientific Computing, LNCC/MCTIC, Petrópolis, Brazil.

Abstract. Glaucoma is the leading cause of irreversible but preventable blindness in the world. Its major treatable risk factor is the intra-ocular pressure, although other biomarkers are being explored to improve the understanding of the pathophysiology of the disease. It has been recently observed that glaucoma induces changes in the ocular hemodynamics. However, its effects on the functional behavior of the retinal arterioles have not been studied yet. In this paper we propose a first approach for characterizing those changes using computational hemodynamics. The retinal blood flow is simulated using a 0D model for a steady, incompressible non Newtonian fluid in rigid domains. The simulation is performed on patient-specific arterial trees extracted from fundus images. We also propose a novel feature representation technique to comprise the outcomes of the simulation stage into a fixed length feature vector that can be used for classification studies. Our experiments on a new database of fundus images show that our approach is able to capture representative changes in the hemodynamics of glaucomatous patients. Code and data are publicly available in https://ignaciorlando.github.io.

## 1 Introduction

Glaucoma is a neurodegenerative condition that is the the leading cause of irreversible but preventable blindness [1].The disease is characterized by the interruption in the communication between the retinal photoreceptors and the brain, preventing visual signals to be carried to the brain via the optic nerve [1]. These progressive changes are asymptomatic until advanced stages of the disease, when undiagnosed patients notice the disease when a considerable amount of visual field has been lost. The major treatable glaucoma risk factor is the intra-ocular pressure (IOP), although individuals with high IOP might not develop glaucoma and subjects with normal IOP values can develop it [2]. Hence, alternative genetic, vascular, anatomical or systemic factors are being extensively studied to better understand the pathophysiology of glaucoma [2,3].

The ocular hemodynamics has long been observed to be affected by glaucoma [2,4] through imaging modalities such as color Doppler imaging or laser

Doppler flowmetry, among others [4]. These tools provide non-invasive measurements of hemodynamic parameters from ocular vascular structures such as the central retinal artery (CRA), the ophthalmic artery (OA) or the retinal capillary network. In turn, the analysis of the hemodynamic characteristics of the retinal arterioles and venules, easily accessible through fundus photography, has been limited despite the availability of this imaging modality for several decades.

In this paper, we propose a first approach to characterize the hemodynamics of the retinal microvasculature of glaucomatous and control subjects based on computer simulations performed on vascular networks extracted from fundus images (Fig. 1). Our method is based on patient-specific graph representations of the arterial network that are used as the topological vascular substrate to build 0D models, which account for the steady, incompressible flow of a non Newtonian fluid (the blood) in rigid domains. Although simpler than fully 3D [5] or 2D [6] blood flow models, 0D models (also known as lumped parameter models) are computationally cheaper and, thus, enable the simulation of blood flow in large networks of vessels [7]. Similar modeling approaches have been used before in the context of retinal hemodynamics simulation. In [7], the blood hematocrit is incorporated as transport species in the model, considering constant crosssectional radius per vessel segment in a network of arteries and veins of a mouse model. More recently, 0D modeling has been applied in [8] for characterizing abnormal changes due to diabetic retinopathy. Although the simulation approach is similar to ours, a Newtonian fluid model was used in [8]. Furthermore, instead of using the whole retinal network, the numerical experiments have been performed in a single vascular bifurcation, which was manually isolated for each image. As a further novel aspect of the present study, the simulation outcomes are summarized in a fixed-length feature vector using a novel technique inspired in the bag of words [9] technique for feature representation. This allows the characterization approach to be applied in combination with linear classifiers to study its relationship with the presence/absence of the disease. We empirically validate this approach on a new database, namely LES-AV, comprising 22 fundus photographs with available manual segmentations of the retinal vessels and their expert classification into arteries and veins.

In summary, the key contributions of this study are: (i) a computational workflow for quantifying patient-specific hemodynamics non-invasively; (ii) a feature representation technique to comprise the simulated outcomes into a fixed length feature vector; (iii) a first pilot study showing that glaucomatous patients exhibit changes in the hemodynamic variables with respect to control subjects; and (iv) a new database of fundus images that is released, jointly with code and implementation details, in https://ignaciorlando.github.io.

## 2 Materials and Methods

The experiments were conducted on a new database of color fundus images, called LES-AV, comprising 22 images/patients with resolutions of 30° FOV and 1444 × 1620 pixels (21 images), and 45° FOV and 1958 × 2196 pixels (one image),

![](images/25cf34c289a31e8cebf11f56f4b1cb846442bb13cbe159c01a2b4a8dca290d75.jpg)  
Fig. 1. Proposed method for characterizing patient-specific retinal hemodynamics

with each pixel = 6 µm. Demographic data about the groups are provided in Section 3 and the supplementary materials.

## 2.1 Preprocessing

Patient-specific segmentations of the retinal vasculature must be first retrieved from a fundus image, and subsequently classified into arteries and veins. Although several methods have been introduced for this purpose [10], we follow a semiautomatic approach to ensure a proper input for the simulation step. Hence, a first coarse segmentation of the vessels is obtained using a fully convolutional neural network [11] (see supplementary materials for details). The segmentation is then manually refined to improve the vessel profiles and their connectivity, and each structure is manually labelled as artery or vein. Subsequently, the arteries are taken and their corresponding centerlines are automatically identified using a skeletonization method [12]. The result is a binary mask $T$ with $N _ { I }$ connected components. Since the vasculature is observed as a 2D projection of the original 3D structure in a fundus image, the vessels in the optic disc (OD) are generally overlapped with each other, making unfeasible the identification of the roots of the trees in any realistic scenario. To mitigate this issue, the arterial segments inside the OD are removed from $T ,$ and a root pixel $I _ { t }$ is automatically chosen for each connected component $T _ { t } \in T$ , such that $I _ { t }$ corresponds to the closest pixel to the OD center. Subsequently, each tree $T _ { t }$ is modeled as a graph structure $G _ { t } = \langle I _ { t } , \mathcal { V } _ { t } , \mathcal { E } _ { t } \rangle$ , where each node $v _ { j } \in \mathcal V _ { t }$ is a set of branching pixels in $T _ { t }$ , and each edge $e _ { ( i , j ) } \in \mathcal { E } _ { t }$ is a segment of centerline pixels connecting two branching points $v _ { i }$ and $v _ { j }$ . The pixels in the skeletonized arterial trees are automatically classified as part of a node or an edge in $G _ { t }$ by analyzing their neighborhood.Finally, the cross-sectional lumen radius at each centerline pixel is estimated from the segmentation using the Euclidean transform [13].

## 2.2 Simulation of the retinal hemodynamics

The simulation is performed on a patient-specific computational mesh associated to the graphs $G _ { t }$ (Section 2.1). This structure comprises the key information required by the model to perform the simulation: (a) the topological connectivity of the arterial segments, through their branching points; (b) the pixel-wise information of the arterial radius, $r _ { i } , ~ i ~ = ~ \{ 1 , . . . , N \}$ , with N the number of centerline pixels (a computational node per pixel is considered); (c) the root points, which are the network inlets $I _ { t } , t = 1 , . . . , N _ { I } ;$ and the network outlets $O _ { m } , m = 1 , . . . , N _ { O }$

A steady state 0D model is used to describe the incompressible flow of a non-Newtonian fluid in non-compliant vessels. Such a model is ideal for this scenario, as it allows to efficiently analyze different simulations, and provides physiologically reasonable coarse descriptions of the hemodynamics in vascular networks. The non Newtonian properties of the blood are incorporated by expressing the viscosity as a function of the vessel cross-sectional radius, to describe the Fahraeus-Lindqvist effect [14]. As a result of the simulation process, the blood pressure (Pi) and the blood flow rate $Q _ { i }$ are computed at each computational node (i). The governing 0D equations are detailed below.

The standard mass conservation and pressure continuity junction model is used at each bifurcation point, where node i branches to nodes $j$ and $k \left( \mathrm { E q . \Gamma ( 1 ) } \right)$ 1

Given an arterial segment of M pixels length, $M - 1$ lumped parameter elements are used, where the mass conservation and the hydraulic analogue of the Ohm's law are considered. Then, for an element formed by nodes i and $i + 1 \ ( \mathrm { E q . \ ( 2 ) } )$ , the resistance to the flow at each segment is modeled with the Poiseuille's law: $R _ { i , i + 1 } = 8 \mu L / \pi r _ { i , i + 1 } ^ { 4 }$ where L is the segment length, $r _ { i , i + 1 } =$ $( r _ { i } + r _ { i + 1 } ) / 2$ is the average radius and $\mu = \mu ( r _ { i , i + 1 } )$ is the blood viscosity, which is assumed to be a function of the arterial radius (since $r < 1 5 0 \mu \mathrm { m } )$ , see the supplementary materials.

$$
\left\{ \begin{array} { l l } { Q _ { i } } & { = Q _ { j } + Q _ { k } , } \\ { P _ { i } } & { = P _ { j } = P _ { k } . } \end{array} \right.\tag{1}
$$

$$
\left\{ \begin{array} { l l } { Q _ { i + 1 } } & { = Q _ { i } , } \\ { Q _ { i + 1 } } & { = \displaystyle \frac { P _ { i } - P _ { i + 1 } } { R _ { i , i + 1 } } . } \end{array} \right.\tag{2}
$$

Regarding boundary conditions, the pressure and flow rates, are prescribed at the inlet and outlets, respectively:

$$
\left\{ \begin{array} { l l } { P _ { I \iota } = P _ { 0 } , } & { \forall l = 1 , \ldots , N _ { I } , } \\ { Q _ { O _ { m } } = \beta r _ { O _ { m } } ^ { \gamma } , } & { \forall m = 1 , \ldots , N _ { O } . } \end{array} \right.\tag{3}
$$

The pressure at all the inlets is set to the mean CRA pressure, $P _ { 0 }$ while the flow at the outlets is set to the value given by Murray's law , which relates the flow rate to the outlet radius to the power of $\gamma .$ Specifically, we set $\gamma = 2 . 6 6 \left[ 1 5 \right]$ 2 and putting $\beta = Q _ { T } / \sum _ { m } r _ { O _ { m } } ^ { \gamma }$ ensures the total retinal flow $Q _ { T }$

The inlet pressure is set to the mean value of normotensive patients (i.e. $P _ { 0 } = 6 2 . 2 2 ~ \mathrm { m m H g } )$ . Due to the lack of consensus on normal values for $Q _ { T } [ 1 6 ]$ we proposed three scenarios, namely $Q _ { T } ^ { \mathrm { S C 1 } } = 3 0 , Q _ { T } ^ { \mathrm { S C 2 } } = 4 5 . 6 , Q _ { T } ^ { \mathrm { S C 3 } } = 8 0 \quad \mu \mathrm { l / m }$ which represent the mean (SC2) and extreme values (SC1 and SC3) for healthy subjects. These values are preset for any given subject, without using patientspecific parameters. Further details about the physiological considerations of the parameters choice are provided in the supplementary materials.

The resulting real linear system of equations is solved using LAPACK dgles function through a QR factorization within a custom C++ source code. After solving the system, for each centerline pixel i in the network G, a feature vector $\mathbf { F } _ { i }$ is computed, comprising: the flow rate $Q _ { i } ,$ the blood pressure $P _ { i } ,$ the mean cross sectional velocity $v _ { i } = Q _ { i } / ( \pi r _ { i } ^ { 2 } )$ , the Poiseuille's resistance $R _ { i }$ , the Reynolds number $\mathrm { R e } _ { i } = \rho 2 r _ { i } v _ { i } / \mu _ { i }$ and the wall shear stress $\mathrm { W S S } _ { i } = { 4 \mu _ { i } Q _ { i } } / ( { \pi r _ { i } ^ { 3 } } )$ . Here, $\rho = 1 . 0 4 0 ~ \mathrm { g / c m ^ { 3 } }$ is the blood density. At bifurcation points, the variables assume the value of the last computational node of the parent artery.

## 2.3 Bag of Hemodynamic Features (BoHF)

One key aspect of the functional characterization is how to construct a feature vector F for discriminating control subjects from diseased. As $\mathbf { F } _ { i }$ is given for every centerline pixel i, it is necessary to find an alternative feature representation with a fixed length, suitable $\mathrm { e . g . }$ for linear discriminant analysis. To this end, we propose a new technique inspired by the Bag of Words (BoW) method [9], named as Bag of Hemodynamic Features (BoHF).

Our analysis is focused on three relevant structures for the graph $G ^ { ( u ) }$ of the u-th subject: the terminal and the bifurcation points, and the arterial segments. For every pixel i being a terminal or a bifurcation point, the vector $\mathbf { F } _ { i }$ is used. However, for a given arterial segment $s \in \mathcal { E }$ , the average vector $\tilde { \mathbf { F } } _ { s }$ over the pixels enclosed by the segment is used as a summary statistic. This allows our representation to lower its variance and to normalize for the length of an arterial segment. The resulting vectors associated to each arterial segment, bifurcation and terminal points are collected into a set, denoted X.

Let $S = \{ ( X ^ { ( u ) } , y ^ { ( u ) } ) \} , u \in \{ 1 , . . . , n \}$ be a training set of n samples, where $X ^ { ( u ) }$ is the set of summary statistics at arterial segments, bifurcation and terminal points described above, and $y ^ { ( u ) } \in \mathcal { L } = \{ - 1 , + 1 \}$ is its associated binary label (e.g. healthy or glaucomatous). Since the size of $\dot { X } ^ { ( u ) }$ varies from one subject to another, our purpose is to map it into a feature vector $\mathbf { x } ^ { ( u ) } \in \mathbb { R } ^ { d }$ , with d fixed. In the traditional BoW approach, the x vectors are histograms of codewords (analogous to words in text documents), and are obtained based on a codebook C (analogous to a word dictionary). C must be designed such that it summarizes the most representative characteristics of each class in ${ \mathcal { L } } .$ Let us denote by $S _ { - 1 }$ and $S _ { + 1 }$ the set of negative and positive samples in $S ,$ respectively. A codebook $\mathcal { C } = \{ \mathcal { C } _ { - 1 } , \mathcal { C } _ { + 1 } \}$ can be automatically learned by applying k-means clustering in the p-dimensional feature space of each subset $S _ { ( \cdot ) }$ . The centroids of each of the learned k clusters can be then taken as the codewords, resulting in a set of k codewords for each class $( | { \mathcal { C } } | = d = 2 k )$ . Hence, the feature vector x can be obtained such that the $j \cdot$ -th position in x corresponds to the number of nearest neighbors in $X ^ { ( u ) }$ to the j-th code in C. If the codewords are representative of each label group, it is expected that the samples belonging to the positive/negative class will have higher/lower values in the last k positions of x.

Finally, the discrimination of glaucomatous patients from control subjects can be posed as a binary classification problem that can be tackled using any binary classification algorithm, e.g. $\ell _ { 2 }$ regularized logistic regression. The linear discriminant function of the logistic regression model $f ( \mathbf { x } ) = \langle \beta , \mathbf { x } \rangle$ with parameters $\beta$ is obtained by solving: $\begin{array} { r } { \hat { \beta } = \arg \operatorname* { m i n } _ { \boldsymbol { \beta } } \left[ \lambda \| \boldsymbol { \beta } \| _ { 2 } ^ { 2 } + \sum _ { u = 0 } ^ { n } \log \bigl ( 1 + e ^ { - y ^ { ( u ) } \langle \boldsymbol { \beta } , \mathbf { x } ^ { ( u ) } \rangle } \bigr ) \right] } \end{array}$

![](images/c4e05c75cc5c47d207e27a3861adf2fa5b05d84817520ab5d84e0016edbbd8a2.jpg)  
(a) Samples

![](images/56bc14d8d9ec40e9adc9edf85f9b7687c97f9629dec63e6ba396cacf37ccc943.jpg)  
(b) Fitted curves  
Fig. 2. Mean radius vs. flow rate per segment for healthy (black) and glaucomatous (red) subjects in the LES-AV database. (a) Samples per image and fitted curves for each individual. (b) Fitted curves for each group. The SC1 and SC3 were used for the glaucomatous and healthy groups, respectively.

## 3 Results

The computational simulations of all scenarios resulted in mean values of pressure drop (∆P to inlet, P0) and blood velocity (v) consistent with previous studies in the human retinal circulation [6].

The ability of the BoHF to capture the hemodynamic changes in the glaucomatous (G) from normal (H) subjects was empirically assessed using $\ell _ { 2 }$ regularized logistic regression on LES-AV, in a leave-one-out cross-validation setting. The SC2 simulations were used for all the cases to avoid any bias in the classification performance. The values of $k \in \{ 2 , . . . , 1 5 \}$ (BoHF length) and $\lambda \in 1 0 ^ { i } , i \in \{ - 3 , . . . , 0 , . . . , 5 \}$ (regularization parameter) were fixed to the values that maximized the accuracy in a held-out validation set randomly sampled from the training set. The features were centered to zero mean and unit variance using the training sets mean and standard deviation. The resulting area under the ROC curve was 0.70248, with an accuracy of 68.18%.

The hemodynamic quantities obtained using the outcomes of SC1 and SC3 for the G and H groups, respectively, are in line with clinical observations. Fig. 2(a) shows the flow vs. mean radius per segment for all subjects. The exponential curves fitted for each group are depicted in Fig. 2(b). The variables are correlated (Spearman's $\rho = 0 . 7 1 , p \ll 0 . 0 1 )$ , which is consistent with the literature [6].

Finally, statistical analysis at a confidence level of 99% were performed on a per-patient and a per-measurement basis. On the former (Table 3), we focused on the age, sex and number of segments (NoS). The differences in the mean values of the NoS and the age were not significant (Wilcoxon test, $p > 0 . 1 )$ Moreover, the patient diagnosis was independent from the sex $( \chi ^ { 2 } \ \mathrm { t e s t } , p = 1 )$

On a per-measurement basis (Table 3), all the arterial segments, terminal and bifurcations points were considered as separate measurements. The analysis was focused on the $\varDelta P , v , r ,$ age and sex. The differences in the mean value of $\varDelta P$ and v were larger in the H group, while the age was larger in the G group $( p \ll 0 . 0 1 )$ (two tailed Wilcoxon test). The r was lower in the G group, although not significant $( p = 0 . 3 1 )$ . As in the per-patient analysis, the diagnostic was independent from the sex $( \chi ^ { 2 }$ test, $p = 0 . 0 2 )$ . The Pearson correlation coefficient between age vs. $\varDelta P$ and age vs. v was close to zero and not significant, suggesting that the age is not an influential factor for the comparison of the means.

<table><tr><td>Variable / Group</td><td> $\mathbf { A l l } \ ( n = 2 2 )$ </td><td>Healthy  $( n = 1 1 )$ </td><td>Glaucoma- tous  $( n = 1 1 )$ </td></tr><tr><td>NoS</td><td> $\overline { { 3 3 . 2 7 \pm 1 3 . 7 2 } }$ </td><td> $\overline { { 3 7 . 9 1 \pm 1 6 . 7 8 } }$ </td><td> $2 8 . 6 4 \pm 8 . 1 5$ </td></tr><tr><td>Age [years]</td><td> $7 1 . 3 6 { \pm } 9 . 9 8 $ </td><td>68.64±9.24</td><td>74.09±10.36</td></tr><tr><td>Sex (Males)</td><td>12</td><td>6</td><td>6</td></tr></table>

Table 1. Summary of statistics at a per-patient level. None of the variables present statistically significant differences between groups $( p \gg 0 . 0 1 )$ 1

<table><tr><td>Variable / Group</td><td>All  $( n = 1 4 6 6 )$ </td><td>Healthy  $( n = 8 3 6 )$ </td><td>Glaucoma- tous  $( n = 6 3 0 )$ </td></tr><tr><td> $\overline { { \varDelta P \ [ m m H g ] ^ { * } } }$ </td><td>6.94±7.40</td><td> $8 . 0 8 { \pm } 8 . 1 2$ </td><td> $\overline { { 5 . 4 3 \pm 5 . 9 9 } }$ </td></tr><tr><td> $v [ c m / s ] ~ ^ { \prime }$ </td><td> $2 . 2 6 { \pm } 1 . 8 4 $ </td><td> $2 . 6 1 { \pm } 2 . 0 3$ </td><td> $1 . 7 9 { \pm } 1 . 4 1 $ </td></tr><tr><td>[r [cm]</td><td> $0 . 0 0 3 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 0 0 3 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 0 0 3 { \scriptstyle \pm 0 . 0 0 1 }$ </td></tr><tr><td> $\mathrm { ~ \textbf ~ { ~ } ~ } \mathrm { ~ A g e ~ } [ \dot { y } e a r s ] ^ { \mathrm { ~ * ~ } }$ </td><td>72.08±9.68</td><td> $7 0 . 8 3 { \pm } 9 . 7 4 $ </td><td>73.74±9.35</td></tr><tr><td>Sex (Males) 2</td><td>763</td><td>413</td><td>350</td></tr></table>

Table 2. Summary of statistics at a per-measurement level (terminals, bifurcations and arterial segments). (\*) points statistically significant differences between groups $( p ~ \ll ~ 0 . 0 1 )$ . (†) reached $p = 0 . 0 2$

## 4 Discussion

We have presented a first approach to characterize the retinal hemodynamics of glaucoma patients using computational fluid-dynamics. To the best of our knowledge, this is the first study in which the relationship between glaucoma and simulated hemodynamic outcomes of the retinal arterioles are studied.

The arterial radius was previously observed to be smaller in glaucoma patients than in normal subjects [17], a setting that is consistent with our data. On the other hand, in-vivo velocity measurements in the CRA of affected patients was shown to be lower than in control subjects [18]. Under these considerations, we have used SC1 for the G group and SC3 for the H group in our first pilot study, resulting in a simulated behavior that is in line with existing observations made on other measurable vessels (i.e. $\mathrm { G } ( v ) < \mathrm { H } ( v ) )$ [18]. Moreover, this rendered physiologically correct values of ∆p in the retinal network.

The classification experiment based on BoHF and the unbiased SC2 outcomes showed an AUC different than random for identifying the glaucomatous patients, indicating that the tool is able to preserve the variations in the hemodynamics of the arteries affected by the disease. Future efforts will be focused on incorporating the venous network and also modeling of the IOP, which is relevant for clinical studies [19]. Also, the incorporation of a deep learning based method for simultaneous segmentation and classification of the vasculature will be explored in order to allow a more efficient characterization of larger populations.

Finally, our MATLAB/C++/python code and the LES-AV database are publicly released. To the best of our knowledge, our data set is the first in providing not only the segmentations of the arterio-venous structures but also diagnostics and clinical parameters at an image level.

## Acknowledgements

This work is funded by ANPCyT PICTs 2016-0116 and start-up 2015-0006, the FWO G0A2716N, WWTF VRG12-009 and a NVIDIA Hardware Grant. JIO is now with OPTIMA, Medical University of Vienna, Austria.

## References

1. Tham, Y.C., et al.: Global prevalence of glaucoma and projections of glaucoma burden through 2040: a systematic review and meta-analysis. Ophthalmology 121(11) (2014) 2081–2090

2. Harris, A., et al.: Ocular hemodynamics and glaucoma: the role of mathematical modeling. Eur J Ophthalmol 139–146

3. Barbosa-Breda, J., et al.: Clinical metabolomics and glaucoma. Ophthalmic research 59(1) (2018) 1–6

4. Abegão Pinto, L., et al.: Ocular blood flow in glaucoma-the Leuven Eye Study Acta Ophthalmologica 94(6) (2016) 592–598

5. Lu, Y., et al.: Computational fluid dynamics assisted characterization of parafoveal hemodynamics in normal and diabetic eyes using adaptive optics scanning laser ophthalmoscopy. Biomed Opt Express 7(12) 4958

6. Liu, D., et al.: Image-based blood flow simulation in the retinal circulation. In: IFMBE Proceedings, Springer (2009) 1963–1966

7. Ganesan, P., He, S., Xu, H.: Analysis of retinal circulation using an image-based network model of retinal vasculature. Microvascular Research 80(1) 99–109

8. Caliva, F., et al.: Hemodynamics in the retinal vasculature during the progression of diabetic retinopathy. JMO 1(4) (2017) 6–15

9. Fei-Fei, L., Perona, P.: A bayesian hierarchical model for learning natural scene categories. In: CVPR. Volume 2., IEEE (2005) 524–531

10. Moccia, S., et al.: Blood vessel segmentation algorithms—review of methods, datasets and evaluation metrics. CMPB 158 (2018) 71–91

11. Giancardo, L., Roberts, K., Zhao, Z.: Representation learning for retinal vasculature embeddings. In: Fetal, Infant and Ophthalmic Medical Image Analysis. Springer (2017) 243–250

12. Rumpf, M., Telea, A.: A continuous skeletonization method based on level sets. In: Eurographics / IEEE VGTC Symposium on Visualization. (2002) 151–159

13. Maurer, C.R., Qi, R., Raghavan, V.: A linear time algorithm for computing exact Euclidean distance transforms of binary images in arbitrary dimensions. IEEE PAMI 25(2) (2003) 265–270

14. Pries, A.R., Secomb, T.W., Gaehtgens, P.: Biophysical aspects of blood flow in the microvasculature. Cardiovasc Res 32(4) 654–667

15. Blanco, P., Queiroz, R., Feijóo, R.: A computational approach to generate concurrent arterial networks in vascular territories. Int J Numer Method Biomed Eng 29 (2013) 601–614

16. Pournaras, C.J., Riva, C.E.: Retinal blood flow evaluation. Ophthalmologica 229(2) (2013) 61–74

17. Mitchell, P., et al.: Retinal vessel diameter and open-angle glaucoma: the Blue Mountains Eye Study. Ophthalmology 112(2) (2005) 245–250

18. Abegão Pinto, L., Vandewalle, E., Stalmans, I.: Disturbed correlation between arterial resistance and pulsatility in glaucoma patients. Acta Ophthalmol 90(3) (2012)

19. Abegão Pinto, $\operatorname { L } , ,$ et al.: Lack of spontaneous venous pulsation: possible risk indicator in normal tension glaucoma? Acta Ophthalmol 91(6) (2013) 514–520