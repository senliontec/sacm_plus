# SACM (official)

- Upstream: https://github.com/kylechuuuuu/SACM (paper code)
- Vendored snapshot: `baselines/sacm` (development reference tree)
- License: no LICENSE file in the vendored snapshot; add the upstream
  license before redistributing this copy.
- Files copied: `segment_anything/` (relative imports; `predictor.py`'s
  absolute `segment_anything.modeling` import rewritten).
- Modifications: added the optional per-block activation-checkpointing
  branch in `modeling/image_encoder.py` for 11 GiB training; it changes
  memory, not the computation.
- Runtime consumer:
  `baseline_zoo.adapters.sacm_official.SACMOfficialAdapter`
