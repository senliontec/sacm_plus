# MaskVSC

- Upstream: https://github.com/Zhouyi-Zura/MaskVSC
- Vendored snapshot: `3rd/MaskVSC`, HEAD
  `5e60565b5ec63cb91295e06b9b502d6350e89d76`.
- License: not present in the vendored snapshot. The upstream README links a
  `LICENSE` file; add it verbatim before redistributing this copy.
- Files copied:
  - `model/UNet.py` -> `unet.py` (unchanged)
  - `lib/Loss.py` -> `loss.py` (only `.cuda()` -> `img.device`)
  - `ImageDataset_Fundus` corruption branch -> `transforms.py`
    (extracted verbatim, plus an empty-skeleton guard)
- Runtime consumer: `baseline_zoo.adapters.maskvsc.MaskVSCAdapter`
