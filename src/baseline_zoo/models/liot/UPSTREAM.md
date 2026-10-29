# LIOT + IterNet upstream record

## Sources

| Component | Upstream | Version |
|---|---|---|
| Paper | "Local Intensity Order Transformation for Robust Curvilinear Object Segmentation", IEEE TIP 2022 | 2022 |
| Code | `https://github.com/TY-Shi/LIOT` (local `3rd/LIOT`) | default branch |
| Paper parsing | `parsing/[2022 tip] Local Intensity Order Transformation for Robust Curvilinear Object Segmentation` | local |
| License | MIT (`3rd/LIOT/LICENSE`) | upstream |

## Vendored files

| Release | Port |
|---|---|
| `src/iternet/unet_parts.py` | `unet_parts.py` (verbatim) |
| `src/iternet/iternet_model.py` | `iternet_model.py` (verbatim) |
| `LIOT.py::distance_weight_binary_pattern_faster` | `liot.py::local_intensity_order` (vectorised NumPy translation; verified bit-for-bit against the release on random and DRIVE images) |

The model is the release's `Iternet(n_channels=4, n_classes=2)` — a U-Net
followed by three mini U-Nets, with 32 base channels and 13,604,360
parameters.  LIOT is the 4-channel local intensity-order representation
used as the network input.

## Protocol adaptations

- **Polarity.**  LIOT assumes the curvilinear object is darker than the
  background.  Retinal/ophthalmic datasets (DRIVE, CHASEDB1, FIVES, HRF,
  LES-AV, OCTA-500, STARE) keep the official dark polarity.  Bright
  structures (CORN, DeepGlobe, ROAD, TopoMortar, WIRE) are inverted before
  LIOT at both training and inference; `polarity_for_dataset` records the
  table and supports `dark`/`bright` overrides.  Unknown datasets default
  to the paper polarity (dark).
- **Budget.**  The release README describes 1000 epochs and the included
  `configuration.txt` uses 170; this port uses the shared 800-epoch project
  budget with validation-Dice selection.
- **Batch.**  Official batch 32; the 18-image split uses batch 8 with
  4-step accumulation, preserving an effective batch of 32.  The release's
  `CurvilinearDataset(repeat=30)` is preserved with a seeded replacement
  sampler (`--liot-repeat 30`), so each epoch exposes 540 random crops.
- **Loss.**  Official cross-entropy is kept.  The optional post-150-epoch
  topological loss in `curvilinearNN_trainin_topo.py` imports a hard-coded
  `/home/sty/...` PersistencePython module and is not part of the LIOT
  representation contribution; it is not used (weight 0 in the release's
  main CE recipe).
- **Augmentation.**  Random 128x128 crops and the release's rotation (±180,
  p=0.5), shear (±5°, p=0.05), horizontal/vertical flip (p=0.5), shift
  (±10%, p=0.05) and zoom (0.8–1.2, p=0.05) are reimplemented with
  torchvision; the release applies the same functional transforms to image
  and mask.
- **Inference.**  The release predicts overlapping 128x128 patches with
  stride 8 and sweeps the F1-optimal threshold.  This port performs a
  whole-image forward pass (the U-Net is fully convolutional), thresholds
  at 0.5 for the shared boolean protocol, and returns the mask at the exact
  GT resolution.  This is a speed/protocol adaptation, not a change to the
  network.
- No official weights are loaded; training starts from random
  initialization.  The release weights are not required for reproduction.

## Verification

- `local_intensity_order` matches the release's
  `distance_weight_binary_pattern_faster` bit-for-bit on random uint8
  images and a DRIVE image (max absolute difference 0).
- Official `Iternet(4, 2)` forward: `[2,4,128,128] -> [2,2,128,128]`,
  13.6M parameters, finite logits.
- Adapter tests cover the polarity table, 4-channel batch preparation,
  CE loss/backward, optimizer/scheduler and GT-resolution prediction.
- GPU smoke (2 train / 1 val, one epoch): train loss 0.9640, val Dice
  0.1596, checkpoint saved; 20 DRIVE predictions took 2 s and exported
  565x584 0/255 masks.
