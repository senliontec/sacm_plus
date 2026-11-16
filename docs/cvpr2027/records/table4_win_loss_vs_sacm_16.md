# Win / tie / loss versus SACM official on the 16-dataset subset

W/T/L = model better / tie / SACM better across 16 datasets (4-seed means). Wilcoxon p is descriptive over 16 datasets; no multiple-comparison correction.

| Model | Dice W/T/L | IoU W/T/L | clDice W/T/L | HD95 W/T/L |
|---|---|---|---|---|
| DCD (fixed3) | 13/0/3 (p=0.013) | 13/0/3 (p=0.016) | 15/0/1 (p<0.001) | 9/0/7 (p=0.782) |
| SAM-OCTA | 7/0/9 (p=0.348) | 7/0/9 (p=0.669) | 9/0/7 (p=0.860) | 6/0/10 (p=0.433) |
| CWSAM | 2/0/14 (p=0.006) | 2/0/14 (p=0.005) | 4/0/12 (p=0.083) | 1/0/15 (p=0.005) |
| nnWNet | 6/0/10 (p=0.044) | 6/0/10 (p=0.074) | 4/0/12 (p=0.011) | 5/0/11 (p=0.051) |
| U-Net | 4/0/12 (p=0.013) | 4/0/12 (p=0.018) | 2/0/14 (p=0.006) | 2/0/14 (p=0.009) |
| MaskVSC | 1/0/15 (p<0.001) | 1/0/15 (p<0.001) | 2/0/14 (p=0.001) | 1/0/15 (p=0.003) |
| CS2Net | 3/0/13 (p=0.006) | 3/0/13 (p=0.005) | 3/0/13 (p=0.003) | 1/0/15 (p=0.004) |
| KnowSAM | 0/0/16 (p<0.001) | 0/0/16 (p<0.001) | 0/0/16 (p<0.001) | 1/0/15 (p=0.003) |
| SegDINO | 0/0/16 (p<0.001) | 0/0/16 (p<0.001) | 0/0/16 (p<0.001) | 3/0/13 (p=0.009) |
| SAM-Med2d | 2/0/14 (p<0.001) | 2/0/14 (p<0.001) | 1/0/15 (p<0.001) | 3/0/13 (p=0.003) |
