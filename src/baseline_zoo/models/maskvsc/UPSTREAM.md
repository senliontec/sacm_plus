# MaskVSC

- Upstream: https://github.com/Zhouyi-Zura/MaskVSC
- Vendored snapshot: `baselines/maskvsc` (development reference tree)
- License: not present in the vendored snapshot. The upstream README links a
  `LICENSE` file; add it verbatim before redistributing this copy.
- Files copied:
  - `model/UNet.py` -> `unet.py` (unchanged)
  - `lib/Loss.py` -> `loss.py` (only `.cuda()` -> `img.device`)
  - `ImageDataset_Fundus` corruption branch -> `transforms.py`
    (extracted verbatim, plus an empty-skeleton guard)
- Runtime consumer: `baseline_zoo.adapters.maskvsc.MaskVSCAdapter`
